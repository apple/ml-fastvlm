//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import SwiftUI
import MLX

@main
struct SceneDescriptorApp: App {
    @StateObject private var appStateManager = AppStateManager()
    @Environment(\.scenePhase) private var scenePhase
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appStateManager)
                .onAppear {
                    appStateManager.handleAppLaunch()
                }
                .onChange(of: scenePhase) { oldPhase, newPhase in
                    appStateManager.handleScenePhaseChange(from: oldPhase, to: newPhase)
                }
        }
    }
}

// MARK: - App State Management
@MainActor
class AppStateManager: ObservableObject {
    @Published var shouldForceRestart = false
    @Published var isRecoveringFromCrash = false
    @Published var crashRecoveryMessage = ""
    
    private let crashDetectionKey = "LastAppTermination"
    private let forceRestartKey = "ForceAppRestart"
    private let appVersionKey = "AppVersion"
    
    init() {
        setupCrashDetection()
    }
    
    private func setupCrashDetection() {
        // Check if we're recovering from a crash or forced restart
        let lastTermination = UserDefaults.standard.string(forKey: crashDetectionKey)
        let forceRestart = UserDefaults.standard.bool(forKey: forceRestartKey)
        let lastVersion = UserDefaults.standard.string(forKey: appVersionKey)
        let currentVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
        
        if forceRestart {
            isRecoveringFromCrash = true
            crashRecoveryMessage = "App was manually restarted due to previous issues"
            UserDefaults.standard.removeObject(forKey: forceRestartKey)
        } else if lastTermination != "clean" {
            isRecoveringFromCrash = true
            crashRecoveryMessage = "App recovered from unexpected termination"
        } else if lastVersion != currentVersion {
            isRecoveringFromCrash = true
            crashRecoveryMessage = "App updated - resetting state"
            UserDefaults.standard.set(currentVersion, forKey: appVersionKey)
        }
        
        // Mark app as starting up (will be marked clean on proper shutdown)
        UserDefaults.standard.set("starting", forKey: crashDetectionKey)
    }
    
    func handleAppLaunch() {
        if isRecoveringFromCrash {
            performCrashRecovery()
        }
        
        // Set up memory pressure monitoring
        setupMemoryPressureMonitoring()
    }
    
    func handleScenePhaseChange(from oldPhase: ScenePhase, to newPhase: ScenePhase) {
        switch newPhase {
        case .active:
            // App became active
            UserDefaults.standard.set("active", forKey: crashDetectionKey)
            
        case .background:
            // App went to background - clean up resources
            cleanupOnBackground()
            UserDefaults.standard.set("background", forKey: crashDetectionKey)
            
        case .inactive:
            // App became inactive
            UserDefaults.standard.set("inactive", forKey: crashDetectionKey)
            
        @unknown default:
            break
        }
    }
    
    private func performCrashRecovery() {
        print("[AppStateManager] Performing crash recovery: \(crashRecoveryMessage)")
        
        // Clear MLX cache
        MLX.GPU.clearCache()
        
        // Clear any persistent state that might be corrupted
        clearCorruptedState()
        
        // Force garbage collection
        Task {
            try? await Task.sleep(for: .milliseconds(100))
            MLX.GPU.clearCache()
        }
        
        // Reset recovery flag after a delay
        Task {
            try? await Task.sleep(for: .seconds(3))
            isRecoveringFromCrash = false
            crashRecoveryMessage = ""
        }
    }
    
    private func clearCorruptedState() {
        // Clear any user defaults that might be corrupted
        let keysToRemove = [
            "LastModelState",
            "LastGenerationState",
            "CachedModelInfo"
        ]
        
        for key in keysToRemove {
            UserDefaults.standard.removeObject(forKey: key)
        }
    }
    
    private func cleanupOnBackground() {
        // Clean up resources when app goes to background
        MLX.GPU.clearCache()
        
        // Post notification for other components to clean up
        NotificationCenter.default.post(name: .appEnteredBackground, object: nil)
    }
    
    private func setupMemoryPressureMonitoring() {
        // Monitor for memory pressure
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleMemoryPressure()
        }
    }
    
    private func handleMemoryPressure() {
        print("[AppStateManager] Memory pressure detected - performing cleanup")
        
        // Clear caches
        MLX.GPU.clearCache()
        
        // Post notification for models to cleanup
        NotificationCenter.default.post(name: .memoryPressureDetected, object: nil)
    }
    
    func forceRestart() {
        // Set flag for next launch
        UserDefaults.standard.set(true, forKey: forceRestartKey)
        UserDefaults.standard.set("force_restart", forKey: crashDetectionKey)
        
        // Clean up current state
        MLX.GPU.clearCache()
        
        // Exit app (will restart on next launch)
        exit(0)
    }
    
    func markCleanShutdown() {
        UserDefaults.standard.set("clean", forKey: crashDetectionKey)
    }
    
    deinit {
        // Just set the flag synchronously
        UserDefaults.standard.set("clean", forKey: crashDetectionKey)
        NotificationCenter.default.removeObserver(self)
    }
}

// MARK: - Notification Extensions
extension Notification.Name {
    static let appEnteredBackground = Notification.Name("AppEnteredBackground")
    static let memoryPressureDetected = Notification.Name("MemoryPressureDetected")
    static let forceAppRestart = Notification.Name("ForceAppRestart")
}
