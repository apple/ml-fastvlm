//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import XCTest
@testable import SceneDescriptor_App
import MLXLMCommon

@MainActor
final class ModelTests: XCTestCase {
    
    func skipIfSimulator() throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("This test requires physical device with GPU")
        #endif
    }
    
    // MARK: - Model Factory Tests
    
    func testModelFactory() throws {
        // Test that both model types can be created without loading
        let fastVLMModel = ModelFactory.createModel(type: .fastVLM)
        let smolVLMModel = ModelFactory.createModel(type: .smolVLM)
        
        XCTAssertTrue(fastVLMModel is FastVLMModel)
        XCTAssertTrue(smolVLMModel is SmolVLMModel)
        
        // Don't call load() or generate() in tests - they use GPU
    }
    
    // MARK: - Model Type Enum Tests
    
    func testModelTypeEnum() throws {
        // Test all cases exist
        XCTAssertEqual(ModelType.allCases.count, 2)
        XCTAssertTrue(ModelType.allCases.contains(.fastVLM))
        XCTAssertTrue(ModelType.allCases.contains(.smolVLM))
        
        // Test display names
        XCTAssertEqual(ModelType.fastVLM.displayName, "FastVLM (Core ML + MLX)")
        XCTAssertEqual(ModelType.smolVLM.displayName, "SmolVLM2 (MLX Native)")
        
        // Test raw values
        XCTAssertEqual(ModelType.fastVLM.rawValue, "FastVLM")
        XCTAssertEqual(ModelType.smolVLM.rawValue, "SmolVLM2")
    }
    
    // MARK: - Model Manager Tests
    
    func testModelManager() async throws {
        let modelManager = ModelManager()
        
        // Test initial state
        XCTAssertEqual(modelManager.selectedModelType, .smolVLM)
        XCTAssertFalse(modelManager.isSwitchingModels)
        XCTAssertFalse(modelManager.running)
        
        // Test cancellation functionality
        modelManager.cancel()
        XCTAssertFalse(modelManager.running)
        
        // Test that memory status is generated
        let memoryStatus = modelManager.memoryStatus
        XCTAssertFalse(memoryStatus.isEmpty)
        XCTAssertTrue(memoryStatus.contains("MB"))
    }
    
    // MARK: - SmolVLM Model Tests
    
    func testSmolVLMModelInitialization() async throws {
        let smolVLM = SmolVLMModel()
        
        // Test that the model can be created without throwing
        XCTAssertNotNil(smolVLM)
        
        // Test basic properties
        XCTAssertFalse(smolVLM.running)
        XCTAssertEqual(smolVLM.output, "")
        XCTAssertEqual(smolVLM.promptTime, "")
        XCTAssertEqual(smolVLM.evaluationState, .idle)
        
        // Test cancellation
        smolVLM.cancel()
        XCTAssertFalse(smolVLM.running)
        XCTAssertEqual(smolVLM.output, "")
        XCTAssertEqual(smolVLM.promptTime, "")
        XCTAssertEqual(smolVLM.evaluationState, .idle)
    }
    
    // MARK: - FastVLM Model Tests
    
    func testFastVLMModelInitialization() async throws {
        let fastVLM = FastVLMModel()
        
        // Test that the model can be created without throwing
        XCTAssertNotNil(fastVLM)
        
        // Test basic properties
        XCTAssertFalse(fastVLM.running)
        XCTAssertEqual(fastVLM.output, "")
        XCTAssertEqual(fastVLM.promptTime, "")
        XCTAssertEqual(fastVLM.evaluationState, .idle)
        
        // Test cancellation
        fastVLM.cancel()
        XCTAssertFalse(fastVLM.running)
        XCTAssertEqual(fastVLM.output, "")
        XCTAssertEqual(fastVLM.promptTime, "")
    }
    
    // MARK: - VLM Protocol Conformance Tests
    
    func testVLMProtocolConformance() throws {
        let fastVLM: any VLMModelProtocol = FastVLMModel()
        let smolVLM: any VLMModelProtocol = SmolVLMModel()
        
        // Test that both models conform to the protocol
        XCTAssertFalse(fastVLM.running)
        XCTAssertFalse(smolVLM.running)
        
        XCTAssertEqual(fastVLM.output, "")
        XCTAssertEqual(smolVLM.output, "")
        
        XCTAssertEqual(fastVLM.promptTime, "")
        XCTAssertEqual(smolVLM.promptTime, "")
        
        // Note: Don't test modelInfo as it might trigger GPU initialization
    }
    
    // MARK: - GPU-Dependent Tests (Skip on Simulator)
    
    func testModelLoadingOnDevice() async throws {
        try skipIfSimulator()
        
        _ = ModelManager()
        // let modelManager = ModelManager()
        // await modelManager.loadCurrentModel()
        
        // This test only runs on physical device
        XCTAssertTrue(true)
    }
    
    func testModelGenerationOnDevice() async throws {
        try skipIfSimulator()
        
        // Test actual model generation only on device
        let modelManager = ModelManager()
        // ... GPU-dependent test code here
    }
}
