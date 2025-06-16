//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import XCTest
@testable import SceneDescriptor_App
import AVFoundation

@MainActor
final class SpeechManagerTests: XCTestCase {
    
    var speechManager: SpeechManager!
    
    override func setUp() {
        super.setUp()
        speechManager = SpeechManager()
    }
    
    override func tearDown() {
        speechManager.stop()
        speechManager = nil
        super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testInitialization() {
        XCTAssertNotNil(speechManager)
        XCTAssertFalse(speechManager.isSpeaking)
        XCTAssertFalse(speechManager.isPaused)
    }
    
    // MARK: - Voice Management Tests
    
    func testAvailableVoices() {
        let voices = speechManager.getAvailableVoices()
        XCTAssertGreaterThan(voices.count, 0, "Should have at least one available voice")
        
        // All voices should be English
        for voice in voices {
            XCTAssertTrue(voice.language.hasPrefix("en"), "Voice language should start with 'en'")
        }
    }
    
    func testVoiceNames() {
        let voices = speechManager.getAvailableVoices()
        let voiceNames = speechManager.getVoiceNames()
        
        XCTAssertEqual(voices.count, voiceNames.count, "Voice count should match voice names count")
        
        for voiceName in voiceNames {
            XCTAssertTrue(voiceName.contains("("), "Voice name should contain language code in parentheses")
            XCTAssertTrue(voiceName.contains(")"), "Voice name should contain language code in parentheses")
        }
    }
    
    // MARK: - Preference Tests
    
    func testSpeechRatePreference() {
        let originalRate = speechManager.preferredSpeechRate
        
        // Test setting rate
        speechManager.preferredSpeechRate = 0.8
        XCTAssertEqual(speechManager.preferredSpeechRate, 0.8, accuracy: 0.01)
        
        // Test rate bounds
        speechManager.preferredSpeechRate = 1.5 // Above normal range
        XCTAssertEqual(speechManager.preferredSpeechRate, 1.5, accuracy: 0.01)
        
        speechManager.preferredSpeechRate = 0.0 // Very slow
        XCTAssertEqual(speechManager.preferredSpeechRate, 0.0, accuracy: 0.01)
        
        // Restore original rate
        speechManager.preferredSpeechRate = originalRate
    }
    
    func testAutoReadPreference() {
        let originalAutoRead = speechManager.autoReadResponses
        
        // Test toggling auto-read
        speechManager.autoReadResponses = true
        XCTAssertTrue(speechManager.autoReadResponses)
        
        speechManager.autoReadResponses = false
        XCTAssertFalse(speechManager.autoReadResponses)
        
        // Restore original setting
        speechManager.autoReadResponses = originalAutoRead
    }
    
    // MARK: - Speech Control Tests
    
    func testSpeechControls() {
        // Test that controls don't crash when not speaking
        speechManager.pause()
        XCTAssertFalse(speechManager.isSpeaking)
        XCTAssertFalse(speechManager.isPaused)
        
        speechManager.resume()
        XCTAssertFalse(speechManager.isSpeaking)
        XCTAssertFalse(speechManager.isPaused)
        
        speechManager.stop()
        XCTAssertFalse(speechManager.isSpeaking)
        XCTAssertFalse(speechManager.isPaused)
    }
    
    func testSpeechWithEmptyText() {
        // Test speaking empty text
        speechManager.speak("")
        
        // Should not start speaking with empty text
        XCTAssertFalse(speechManager.isSpeaking)
    }
    
    func testAutoSpeakWhenDisabled() {
        speechManager.autoReadResponses = false
        
        // Should not speak when auto-read is disabled
        speechManager.speakIfAutoEnabled("Test message")
        XCTAssertFalse(speechManager.isSpeaking)
    }
}
