//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import XCTest
@testable import SceneDescriptor_App
import MLXLMCommon
import CoreImage

@MainActor
class ModelGenerationTests: XCTestCase {
    
    var modelManager: ModelManager!
    
    override func setUp() async throws {
        try await super.setUp()
        
        #if targetEnvironment(simulator)
        throw XCTSkip("ModelGenerationTests require physical device with GPU")
        #endif
        
        modelManager = ModelManager()
    }
    
    override func tearDown() {
        modelManager?.cancel()
        modelManager = nil
        super.tearDown()
    }
    
    // MARK: - User Input Creation Tests (Safe for Simulator)
    
    func testUserInputWithTextPrompt() {
        let userInput = UserInput(prompt: .text("Describe this image"), images: [])
        
        // Test that UserInput can be created
        XCTAssertNotNil(userInput)
        
        // Test prompt extraction
        switch userInput.prompt {
        case .text(let text):
            XCTAssertEqual(text, "Describe this image")
        case .messages(_):
            XCTFail("Expected text prompt, got messages")
        case .chat(_):
            XCTFail("Expected text prompt, got chat")
        @unknown default:
            XCTFail("Unknown prompt type")
        }
        
        XCTAssertEqual(userInput.images.count, 0)
        XCTAssertEqual(userInput.videos.count, 0)
    }
    
    func testUserInputWithMessages() {
        let messages = [
            ["role": "user", "content": "What do you see?"]
        ]
        let userInput = UserInput(messages: messages)
        
        XCTAssertNotNil(userInput)
        
        switch userInput.prompt {
        case .messages(let msgs):
            XCTAssertEqual(msgs.count, 1)
            if let firstMessage = msgs.first {
                XCTAssertEqual(firstMessage["role"] as? String, "user")
                XCTAssertEqual(firstMessage["content"] as? String, "What do you see?")
            }
        case .text(_):
            XCTFail("Expected messages prompt, got text")
        case .chat(_):
            XCTFail("Expected messages prompt, got chat")
        @unknown default:
            XCTFail("Unknown prompt type")
        }
    }
    
    func testUserInputWithImage() {
        // Create a test image
        let testImage = CIImage(color: CIColor.blue).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))
        let userInput = UserInput(
            prompt: .text("What color is this?"),
            images: [.ciImage(testImage)]
        )
        
        XCTAssertNotNil(userInput)
        XCTAssertEqual(userInput.images.count, 1)
        
        // Test prompt extraction
        switch userInput.prompt {
        case .text(let text):
            XCTAssertEqual(text, "What color is this?")
        case .messages(_):
            XCTFail("Expected text prompt, got messages")
        case .chat(_):
            XCTFail("Expected text prompt, got chat")
        @unknown default:
            XCTFail("Unknown prompt type")
        }
        
        // Test image type
        switch userInput.images[0] {
        case .ciImage(let ciImage):
            XCTAssertEqual(ciImage.extent.size.width, 100)
            XCTAssertEqual(ciImage.extent.size.height, 100)
        case .url(_):
            XCTFail("Expected CIImage, got URL")
        case .array(_):
            XCTFail("Expected CIImage, got array")
        @unknown default:
            XCTFail("Unknown image type")
        }
    }
    
    func testUserInputWithMultipleImages() {
        let image1 = CIImage(color: CIColor.red).cropped(to: CGRect(x: 0, y: 0, width: 50, height: 50))
        let image2 = CIImage(color: CIColor.green).cropped(to: CGRect(x: 0, y: 0, width: 75, height: 75))
        
        let userInput = UserInput(
            prompt: .text("Compare these images"),
            images: [.ciImage(image1), .ciImage(image2)]
        )
        
        XCTAssertEqual(userInput.images.count, 2)
        
        // Test prompt extraction
        switch userInput.prompt {
        case .text(let text):
            XCTAssertEqual(text, "Compare these images")
        case .messages(_):
            XCTFail("Expected text prompt, got messages")
        case .chat(_):
            XCTFail("Expected text prompt, got chat")
        @unknown default:
            XCTFail("Unknown prompt type")
        }
    }
    
    // MARK: - Model State Tests
    
    func testModelInitialState() {
        let fastVLM = FastVLMModel()
        let smolVLM = SmolVLMModel()
        
        // Test FastVLM initial state
        XCTAssertFalse(fastVLM.running)
        XCTAssertEqual(fastVLM.output, "")
        XCTAssertEqual(fastVLM.promptTime, "")
        XCTAssertEqual(fastVLM.evaluationState, .idle)
        
        // Test SmolVLM initial state
        XCTAssertFalse(smolVLM.running)
        XCTAssertEqual(smolVLM.output, "")
        XCTAssertEqual(smolVLM.promptTime, "")
        XCTAssertEqual(smolVLM.evaluationState, .idle)
    }
    
    func testModelStateAfterCancel() {
        let fastVLM = FastVLMModel()
        let smolVLM = SmolVLMModel()
        
        // Set some state to test cancellation reset
        fastVLM.output = "Some output"
        fastVLM.promptTime = "100 ms"
        
        smolVLM.output = "Some output"
        smolVLM.promptTime = "150 ms"
        
        // Cancel and test reset
        fastVLM.cancel()
        smolVLM.cancel()
        
        XCTAssertFalse(fastVLM.running)
        XCTAssertEqual(fastVLM.output, "")
        XCTAssertEqual(fastVLM.promptTime, "")
        
        XCTAssertFalse(smolVLM.running)
        XCTAssertEqual(smolVLM.output, "")
        XCTAssertEqual(smolVLM.promptTime, "")
    }
    
    // MARK: - Model Manager Generation Tests
    
    func testModelManagerGeneration() async throws {
        // Create a simple test input
        let testImage = CIImage(color: CIColor.white).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))
        let userInput = UserInput(
            prompt: .text("Describe this image"),
            images: [.ciImage(testImage)]
        )
        
        // Test that generate doesn't crash
        let task = await modelManager.generate(userInput)
        
        // Cancel immediately to avoid long-running test
        modelManager.cancel()
        
        // Wait for task completion
        await task.value
        
        // Test should complete without throwing
        XCTAssertTrue(true)
    }
    
    func testModelManagerConcurrentGeneration() async throws {
        let testImage = CIImage(color: CIColor.gray).cropped(to: CGRect(x: 0, y: 0, width: 50, height: 50))
        let userInput = UserInput(
            prompt: .text("Test"),
            images: [.ciImage(testImage)]
        )
        
        // Start first generation
        let task1 = await modelManager.generate(userInput)
        
        // Start second generation (should cancel first)
        let task2 = await modelManager.generate(userInput)
        
        // Cancel both
        modelManager.cancel()
        
        await task1.value
        await task2.value
        
        XCTAssertFalse(modelManager.running)
    }
    
    // MARK: - Image Processing Tests
    
    func testImageTypes() {
        // Test different image types don't crash UserInput creation
        let ciImage = CIImage(color: CIColor.black).cropped(to: CGRect(x: 0, y: 0, width: 10, height: 10))
        let url = URL(string: "https://example.com/image.jpg")!
        
        let userInputCIImage = UserInput(
            prompt: .text("Test"),
            images: [.ciImage(ciImage)]
        )
        
        let userInputURL = UserInput(
            prompt: .text("Test"),
            images: [.url(url)]
        )
        
        XCTAssertNotNil(userInputCIImage)
        XCTAssertNotNil(userInputURL)
        XCTAssertEqual(userInputCIImage.images.count, 1)
        XCTAssertEqual(userInputURL.images.count, 1)
        
        // Test prompt extraction
        switch userInputCIImage.prompt {
        case .text(let text):
            XCTAssertEqual(text, "Test")
        case .messages(_):
            XCTFail("Expected text prompt, got messages")
        case .chat(_):
            XCTFail("Expected text prompt, got chat")
        @unknown default:
            XCTFail("Unknown prompt type")
        }
        
        switch userInputURL.prompt {
        case .text(let text):
            XCTAssertEqual(text, "Test")
        case .messages(_):
            XCTFail("Expected text prompt, got messages")
        case .chat(_):
            XCTFail("Expected text prompt, got chat")
        @unknown default:
            XCTFail("Unknown prompt type")
        }
    }
    
    // MARK: - Error Handling Tests
    
    func testModelGenerationWithoutImages() async throws {
        let userInput = UserInput(prompt: .text("Hello"), images: [])
        
        // Test generation with no images
        let task = await modelManager.generate(userInput)
        modelManager.cancel()
        await task.value
        
        // Should not crash
        XCTAssertTrue(true)
    }
    
    func testModelGenerationWithoutPrompt() async throws {
        let testImage = CIImage(color: CIColor.yellow).cropped(to: CGRect(x: 0, y: 0, width: 10, height: 10))
        let userInput = UserInput(prompt: .text(""), images: [.ciImage(testImage)])
        
        // Test generation with empty prompt
        let task = await modelManager.generate(userInput)
        modelManager.cancel()
        await task.value
        
        // Should not crash
        XCTAssertTrue(true)
    }
    
    // MARK: - Performance Tests
    
    func testUserInputCreationPerformance() {
        let testImage = CIImage(color: CIColor.magenta).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))
        
        measure {
            for i in 0..<100 {
                let userInput = UserInput(
                    prompt: .text("Test prompt \(i)"),
                    images: [.ciImage(testImage)]
                )
                XCTAssertNotNil(userInput)
                
                // Test prompt extraction
                switch userInput.prompt {
                case .text(let text):
                    XCTAssertEqual(text, "Test prompt \(i)")
                case .messages(_):
                    XCTFail("Expected text prompt, got messages")
                case .chat(_):
                    XCTFail("Expected text prompt, got chat")
                @unknown default:
                    XCTFail("Unknown prompt type")
                }
            }
        }
    }
    
    func testModelSwitchingPerformance() async throws {
        measure {
            Task { @MainActor in
                for _ in 0..<5 {
                    await modelManager.switchModel(to: .fastVLM)
                    await modelManager.switchModel(to: .smolVLM)
                }
            }
        }
    }
}
