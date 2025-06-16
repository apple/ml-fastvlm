//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import XCTest
@testable import SceneDescriptor_App

@MainActor
final class MemoryMonitorTests: XCTestCase {
    
    var memoryMonitor: MemoryMonitor!
    
    override func setUp() {
        super.setUp()
        memoryMonitor = MemoryMonitor()
    }
    
    override func tearDown() {
        memoryMonitor?.stopMonitoring()
        memoryMonitor = nil
        super.tearDown()
    }
    
    // MARK: - Initialization Tests
    
    func testInitialization() {
        XCTAssertNotNil(memoryMonitor)
        XCTAssertFalse(memoryMonitor.isMonitoring)
        XCTAssertEqual(memoryMonitor.currentMemoryUsage, 0)
        XCTAssertEqual(memoryMonitor.peakMemoryUsage, 0)
    }
    
    // MARK: - Monitoring Control Tests
    
    func testStartMonitoring() async throws {
        XCTAssertFalse(memoryMonitor.isMonitoring)
        
        memoryMonitor.startMonitoring()
        XCTAssertTrue(memoryMonitor.isMonitoring)
        
        // Wait for monitoring to collect data
        try await Task.sleep(for: .milliseconds(100))
        
        // Should have collected some memory data
        XCTAssertGreaterThan(memoryMonitor.currentMemoryUsage, 0)
        XCTAssertGreaterThan(memoryMonitor.peakMemoryUsage, 0)
    }
    
    func testStopMonitoring() async throws {
        memoryMonitor.startMonitoring()
        XCTAssertTrue(memoryMonitor.isMonitoring)
        
        memoryMonitor.stopMonitoring()
        XCTAssertFalse(memoryMonitor.isMonitoring)
    }
    
    // MARK: - Memory Data Tests
    
    func testMemoryDataCollection() async throws {
        memoryMonitor.startMonitoring()
        
        // Wait for data collection
        try await Task.sleep(for: .milliseconds(200))
        
        // Test that memory data is reasonable
        XCTAssertGreaterThan(memoryMonitor.currentMemoryUsage, 0)
        XCTAssertGreaterThan(memoryMonitor.memoryUsageMB, 0)
        XCTAssertLessThan(memoryMonitor.memoryUsageMB, 16 * 1024) // Less than 16GB
        
        // Peak should be at least as large as current
        XCTAssertGreaterThanOrEqual(memoryMonitor.peakMemoryUsage, memoryMonitor.currentMemoryUsage)
        XCTAssertGreaterThanOrEqual(memoryMonitor.peakMemoryUsageMB, memoryMonitor.memoryUsageMB)
    }
    
    // MARK: - Memory Pressure Tests
    
    func testMemoryPressureLevels() async throws {
        memoryMonitor.startMonitoring()
        
        // Wait for data collection
        try await Task.sleep(for: .milliseconds(100))
        
        let pressureLevel = memoryMonitor.memoryPressureLevel
        let validLevels = ["Low", "Medium", "High", "Critical"]
        
        XCTAssertTrue(validLevels.contains(pressureLevel), "Pressure level '\(pressureLevel)' should be one of: \(validLevels)")
    }
    
    // MARK: - Memory Conversion Tests
    
    func testMemoryUnitConversions() {
        // Test MB conversion
        memoryMonitor.currentMemoryUsage = 1024 * 1024 // 1MB
        XCTAssertEqual(memoryMonitor.memoryUsageMB, 1.0, accuracy: 0.01)
        
        memoryMonitor.peakMemoryUsage = 2 * 1024 * 1024 // 2MB
        XCTAssertEqual(memoryMonitor.peakMemoryUsageMB, 2.0, accuracy: 0.01)
        
        // Test with zero
        memoryMonitor.currentMemoryUsage = 0
        XCTAssertEqual(memoryMonitor.memoryUsageMB, 0.0)
        
        // Test with fractional MB
        memoryMonitor.currentMemoryUsage = 1536 * 1024 // 1.5MB
        XCTAssertEqual(memoryMonitor.memoryUsageMB, 1.5, accuracy: 0.01)
    }
}