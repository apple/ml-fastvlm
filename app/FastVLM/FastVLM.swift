//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import CoreImage
import CoreML
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

// FastVLM is Qwen2VL with a custom vision tower.

// MARK: - Common

/// Rotates half the hidden dims of the input
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let index = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0 ..< index]
    let x2 = x[.ellipsis, index...]
    return concatenated([-x2, x1], axis: -1)
}

// MARK: - Language

private enum Language {

    /// Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors
    static private func applyMultimodalRotaryPositionEmbedding(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray,
        positionIds: MLXArray, mropeSection: [Int]
    ) -> (MLXArray, MLXArray) {
        var cos = cos[positionIds]
        var sin = sin[positionIds]

        cos =
            concatenated(
                // [m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))]
                split(cos, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        sin =
            concatenated(
                split(sin, indices: mropeSection, axis: -1).enumerated().map { i, m in m[i % 3] },
                axis: -1
            )[0..., .newAxis, 0..., 0...]

        // Apply rotary embedding
        let qEmbed = (q * cos) + (rotateHalf(q) * sin)
        let kEmbed = (k * cos) + (rotateHalf(k) * sin)
        return (qEmbed, kEmbed)
    }

    fileprivate class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let mropeSection: [Int]

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        public init(_ args: FastVLMConfiguration.TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            if let v = args.ropeScaling?["mrope_section"], let array = v.asInts() {
                // mrope_section = np.cumsum(mrope_section * 2)[:-1].tolist()
                self.mropeSection = sequence(state: (0, array.makeIterator())) { state in
                    if let v = state.1.next() {
                        // note the *2
                        state.0 += v * 2
                        return state.0
                    } else {
                        return nil
                    }
                }.dropLast()
            } else {
                fatalError("rope_scaling['mrope_section'] must be an array of integers")
            }

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            // prepare the queries, keys and values for the attention computation
            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0

            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

            if let cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: nil
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class FastVLMDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: FastVLMConfiguration.TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    fileprivate class Qwen2Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [FastVLMDecoderLayer]
        fileprivate let norm: RMSNorm

        public init(_ args: FastVLMConfiguration.TextConfiguration) {
            precondition(args.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

            self.layers = (0 ..< args.hiddenLayers)
                .map { _ in
                    FastVLMDecoderLayer(args)
                }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            // Remove mask creation if it's causing issues with the new API
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: nil, cache: cache?[i])
            }

            return norm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Qwen2Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]

        public init(_ args: FastVLMConfiguration.TextConfiguration) {
            self.model = Qwen2Model(args)

            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
            }

            self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        }

        public func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

private enum Vision {

    fileprivate class VisionModelCoreML {

        let lock = NSLock()
        var _model: fastvithd?

        init() {
        }

        func load() throws -> fastvithd {
            try lock.withLock {
                if let model = _model { return model }
                let model = try fastvithd()
                _model = model
                return model
            }
        }

        public func model() -> fastvithd {
            try! load()
        }

        public func encode(_ image: MLXArray) -> MLXArray {
            // MLMultiArray requires mutable input data
            var (data, strides) = {
                let arrayData = image.asType(.float32).asData(access: .noCopyIfContiguous)
                return (arrayData.data, arrayData.strides)
            }()

            precondition(image.ndim == 4)
            precondition(image.dim(0) == 1)
            precondition(image.dim(1) == 3)

            let h = NSNumber(value: image.dim(2))
            let w = NSNumber(value: image.dim(3))

            return data.withUnsafeMutableBytes { (ptr: UnsafeMutableRawBufferPointer) in
                // wrap the backing of the MLXArray
                let array = try! MLMultiArray(
                    dataPointer: ptr.baseAddress!, shape: [1, 3, h, w], dataType: .float32,
                    strides: strides.map { .init(value: $0) })

                // inference
                let output = try! model().prediction(images: array)
                precondition(output.image_features.shape == [1, 256, 3072])
                precondition(output.image_features.dataType == .float32)
                return output.image_features.withUnsafeBytes { ptr in
                    MLXArray(ptr, [1, 256, 3072], type: Float32.self)
                }
            }
        }
    }

    fileprivate class VisionModel: Module {

        let model = VisionModelCoreML()

        public override init() {}

        public func callAsFunction(_ hiddenStates: MLXArray, gridThw: [THW]) -> MLXArray {
            model.encode(hiddenStates)
        }
    }
}

// MARK: - Processor

/// FastVLM `UserInputProcessor`.
///
/// This is meant to be used with ``FastVLM`` and is typically created by ``VLMModelFactory``.
public class FastVLMProcessor: UserInputProcessor {

    private let config: FastVLMProcessorConfiguration
    private let imageProcessingConfig: FastVLMPreProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: FastVLMPreProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = FastVLMProcessorConfiguration()
        self.imageProcessingConfig = config
        self.tokenizer = tokenizer
    }

    public func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // first apply the user requested resizing, etc. if any
        var image = MediaProcessingExtensions.apply(image, processing: processing)

        // image_processing_clip.py
        let size = MediaProcessingExtensions.fitIn(
            image.extent.size, shortestEdge: imageProcessingConfig.size.shortestEdge)
        image = MediaProcessingExtensions.resampleBicubic(image, to: size)

        image = MediaProcessingExtensions.centerCrop(
            image, size: imageProcessingConfig.cropSize.size)

        image = MediaProcessing.normalize(
            image, mean: imageProcessingConfig.imageMeanTuple,
            std: imageProcessingConfig.imageStdTuple)

        let array = MediaProcessingExtensions.asPlanarMLXArray(image)
        return (array, .init(0, array.dim(2), array.dim(3)))
    }

    public func prepare(prompt: UserInput.Prompt, imageTHW: THW?) -> String {
        // Convert prompt to messages based on the new API
        var messages: [[String: Any]]
        switch prompt {
        case .text(let text):
            messages = [["role": "user", "content": text]]
        case .messages(let messageArray):
            messages = messageArray
        @unknown default:
            // Handle any other cases by treating as empty user message
            messages = [["role": "user", "content": ""]]
        }
        
        if let firstMessage = messages.first,
           firstMessage["role"] as? String != "system" {
            messages.insert(["role": "system", "content": "You are a helpful assistant."], at: 0)
        }

        let lastIndex = messages.count - 1
        var lastMessage = (messages[lastIndex]["content"] as? String) ?? ""

        // processing_llava.py
        if let imageTHW {
            let height = imageTHW.h
            let width = imageTHW.w
            let patchSize = config.patchSize

            var numImageTokens =
                (height / patchSize) * (width / patchSize) + config.numAdditionalImageTokens

            if config.visionFeatureSelectStrategy == .default {
                numImageTokens -= 1
            }

            lastMessage += Array(repeating: config.imageToken, count: numImageTokens)
                .joined()
        }

        messages[lastIndex]["content"] = lastMessage

        return
            messages
            .map { message in
                let role = (message["role"] as? String) ?? "user"
                let content = (message["content"] as? String) ?? ""
                return "<|im_start|>\(role)\n\(content)<|im_end|>"
            }
            .joined(separator: "\n")
            + "\n<|im_start|>assistant\n"
    }

    public func prepare(input: UserInput) throws -> LMInput {
        if input.images.isEmpty {
            // just a straight text prompt
            let prompt = prepare(prompt: input.prompt, imageTHW: nil)
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }

        if input.images.count > 1 {
            throw VLMError.singleImageAllowed
        }

        let (pixels, thw) = try preprocess(
            image: input.images[0].asCIImage(), processing: input.processing)
        let image = LMInput.ProcessedImage(pixels: pixels)

        let prompt = prepare(prompt: input.prompt, imageTHW: thw)
        let promptTokens = tokenizer.encode(text: prompt)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(text: .init(tokens: promptArray, mask: mask), image: image)
    }

}

// MARK: - Model

private class FastVLMMultiModalProjector: Module, UnaryLayer {

    @ModuleInfo(key: "linear_0") var linear0: Linear
    @ModuleInfo(key: "gelu") var gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    public init(_ config: FastVLMConfiguration) {
        self._linear0.wrappedValue = Linear(
            config.visionConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
            bias: true)
        self._gelu.wrappedValue = GELU()
        self._linear2.wrappedValue = Linear(
            config.textConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
            bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = linear0(x)
        x = gelu(x)
        x = linear2(x)
        return x
    }
}

/// FastVLM
///
/// This is typically created by ``VLMModelFactory``.
public class FastVLM: Module, VLMModel, KVCacheDimensionProvider {

    static public var modelConfiguration: ModelConfiguration {
        let bundle = Bundle(for: FastVLM.self)
        let url = bundle.url(forResource: "config", withExtension: "json")!
            .resolvingSymlinksInPath()
            .deletingLastPathComponent()
        return ModelConfiguration(directory: url)
    }

    static public func register(modelFactory: VLMModelFactory) {
        modelFactory.typeRegistry.registerModelType("llava_qwen2") { url in
            let configuration = try JSONDecoder().decode(
                FastVLMConfiguration.self, from: Data(contentsOf: url))
            return FastVLM(configuration)
        }

        modelFactory.processorRegistry.registerProcessorType("LlavaProcessor") { url, tokenizer in
            let configuration = try JSONDecoder().decode(
                FastVLMPreProcessorConfiguration.self, from: Data(contentsOf: url))
            return FastVLMProcessor(configuration, tokenizer: tokenizer)
        }
    }

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        FastVLMMultiModalProjector

    public let config: FastVLMConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public init(_ config: FastVLMConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel()
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
        self._multiModalProjector.wrappedValue = FastVLMMultiModalProjector(config)
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?, gridThw: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let gridThw else {
            return languageModel(inputIds).logits
        }

        // Get the input embeddings from the language model
        let inputEmbeds = languageModel.model.embedTokens(inputIds)

        // Get the ouptut hidden states from the vision model
        let imageFeaturesCoreML = self.visionModel(pixelValues, gridThw: gridThw)
        let imageFeatures = multiModalProjector(imageFeaturesCoreML)

        // Insert special image tokens in the input_ids
        return mergeInputIdsWithImageFeatures(
            inputIds: inputIds, inputEmbeds: inputEmbeds, imageFeatures: imageFeatures)
    }

    private func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray, inputEmbeds: MLXArray, imageFeatures: MLXArray
    ) -> MLXArray {
        let imageTokenIndex = config.baseConfiguration.imageTokenId

        var imageIndices = [Int]()
        for (i, v) in inputIds.asArray(Int.self).enumerated() {
            if v == imageTokenIndex {
                imageIndices.append(i)
            }
        }

        inputEmbeds[0..., MLXArray(imageIndices), 0...] = imageFeatures
        return inputEmbeds
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        // Instead of using input.image?.imageGridThw, we need to extract the grid info differently
        // Based on the new API, we might need to store this information separately
        let gridThw: [THW]? = input.image != nil ? [THW(0, input.image!.pixels.dim(2), input.image!.pixels.dim(3))] : nil

        let dtype = DType.float32
        let pixels = input.image?.pixels.asType(dtype)

        let inputEmbeddings = self.inputEmbeddings(
            inputIds: input.text.tokens, pixelValues: pixels, gridThw: gridThw)

        let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbeddings)

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        _ = try? visionModel.model.load()

        return weights
    }
}

// MARK: - Configuration

/// Configuration for ``FastVLM``
public struct FastVLMConfiguration: Codable, Sendable {

    public struct VisionConfiguration: Codable, Sendable {
        public let hiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "mm_hidden_size"
        }
    }

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
        public let vocabularySize: Int
        public let kvHeads: Int
        private let _maxPositionEmbeddings: Int?
        public var maxpPositionEmbeddings: Int { _maxPositionEmbeddings ?? 32768 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 1_000_000 }
        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool { _ropeTraditional ?? false }
        public let _ropeScaling: [String: StringOrNumber]?
        public var ropeScaling: [String: StringOrNumber]? {
            _ropeScaling ?? ["mrope_section": .ints([2, 1, 1])]
        }
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case _ropeScaling = "rope_scaling"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let imageTokenId: Int
        public let hiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case imageTokenId = "image_token_index"
            case hiddenSize = "hidden_size"
        }
    }

    public let visionConfiguration: VisionConfiguration
    public let textConfiguration: TextConfiguration
    public let baseConfiguration: BaseConfiguration

    public init(from decoder: any Swift.Decoder) throws {
        // these are overlaid in the top level
        self.visionConfiguration = try VisionConfiguration(from: decoder)
        self.textConfiguration = try TextConfiguration(from: decoder)
        self.baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

/// Configuration for ``FastVLMProcessor``
public struct FastVLMPreProcessorConfiguration: Codable, Sendable {

    public struct CropSize: Codable, Sendable {
        let width: Int
        let height: Int

        var size: CGSize { .init(width: CGFloat(width), height: CGFloat(height)) }
    }

    public struct Size: Codable, Sendable {
        let shortestEdge: Int

        enum CodingKeys: String, CodingKey {
            case shortestEdge = "shortest_edge"
        }
    }

    public var imageMean: [CGFloat]
    public var imageStd: [CGFloat]
    public var size: Size
    public var cropSize: CropSize

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case cropSize = "crop_size"
    }
}

public struct FastVLMProcessorConfiguration: Codable, Sendable {

    public enum Strategy: Codable, Sendable {
        case `default`
    }

    public var imageToken = "<image>"
    public var numAdditionalImageTokens = 0
    public var patchSize = 64
    public var visionFeatureSelectStrategy: Strategy?

}
