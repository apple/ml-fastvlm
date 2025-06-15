//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import AVFoundation
import CoreImage

#if os(iOS)
    import UIKit
#endif

@Observable
public class CameraController: NSObject {

    private var framesContinuation: AsyncStream<CMSampleBuffer>.Continuation?

    public var backCamera = true {
        didSet {
            Task {
                try? await Task.sleep(for: .milliseconds(100))
                stop()
                start()
            }
        }
    }

    public var devices = [AVCaptureDevice]()

    public var device: AVCaptureDevice? = AVCaptureDevice.default(for: .video) {
        didSet {
            Task {
                try? await Task.sleep(for: .milliseconds(100))
                stop()
                start()
            }
        }
    }

    private var permissionGranted = true
    private var captureSession: AVCaptureSession?
    private let sessionQueue = DispatchQueue(label: "sessionQueue")
    @objc dynamic private var rotationCoordinator : AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?
    
    private var lastFrameTime: CFAbsoluteTime = 0
    private let frameThrottleInterval: CFAbsoluteTime = 1.0 / 30.0 // 30 FPS max

    public func attach(continuation: AsyncStream<CMSampleBuffer>.Continuation) {
        sessionQueue.async {
            self.framesContinuation = continuation
        }
    }

    public func detatch() {
        sessionQueue.async {
            self.framesContinuation?.finish()
            self.framesContinuation = nil
        }
    }

    public func stop() {
        sessionQueue.sync { [self] in
            rotationObservation?.invalidate()
            rotationObservation = nil
            rotationCoordinator = nil
            
            captureSession?.stopRunning()
            captureSession = nil
        }
    }

    public func start() {
        sessionQueue.async { [self] in
            let captureSession = AVCaptureSession()
            self.captureSession = captureSession

            self.checkPermission()
            self.setupCaptureSession(position: backCamera ? .back : .front)
            captureSession.startRunning()
        }
    }

    #if os(iOS)
        private func setOrientation(_ orientation: UIDeviceOrientation) {
            guard let captureSession else { return }

            let angle: Double?
            switch orientation {
            case .unknown, .faceDown:
                angle = nil
            case .portrait, .faceUp:
                angle = 90
            case .portraitUpsideDown:
                angle = 270
            case .landscapeLeft:
                angle = 0
            case .landscapeRight:
                angle = 180
            @unknown default:
                angle = nil
            }

            if let angle {
                for output in captureSession.outputs {
                    output.connection(with: .video)?.videoRotationAngle = angle
                }
            }
        }
    
    private func updateRotation(rotation : CGFloat) {
        guard let captureSession else { return }
        for output in captureSession.outputs {
            output.connection(with: .video)?.videoRotationAngle = rotation
        }
    }
    #endif

    func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            // The user has previously granted access to the camera.
            self.permissionGranted = true

        case .notDetermined:
            // The user has not yet been asked for camera access.
            self.requestPermission()

        // Combine the two other cases into the default case
        default:
            self.permissionGranted = false
        }
    }

    func requestPermission() {
        // Strong reference not a problem here but might become one in the future.
        AVCaptureDevice.requestAccess(for: .video) { [unowned self] granted in
            self.permissionGranted = granted
        }
    }

    func setupCaptureSession(position: AVCaptureDevice.Position) {
        guard let captureSession else { return }

        let videoOutput = AVCaptureVideoDataOutput()

        guard permissionGranted else {
            print("No permission for camera")
            return
        }

        let deviceTypes: [AVCaptureDevice.DeviceType]
        #if os(iOS)
            deviceTypes = [.builtInDualCamera, .builtInWideAngleCamera]
        #else
            deviceTypes = [.external, .continuityCamera, .builtInWideAngleCamera]
        #endif

        let videoDeviceDiscoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes,
            mediaType: .video,
            position: position)

        let videoDevice: AVCaptureDevice?
        if let device = self.device, videoDeviceDiscoverySession.devices.contains(device) {
            videoDevice = device
        } else {
            videoDevice = videoDeviceDiscoverySession.devices.first
        }

        if devices.isEmpty {
            self.devices = videoDeviceDiscoverySession.devices
        }

        guard
            let videoDevice
        else {
            print("Unable to find video device")
            return
        }
        guard let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice) else {
            print("Unable to create AVCaptureDeviceInput")
            return
        }
        guard captureSession.canAddInput(videoDeviceInput) else {
            print("Unable to add input")
            return
        }
        captureSession.addInput(videoDeviceInput)

        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "sampleBufferQueue"))
        captureSession.addOutput(videoOutput)
        captureSession.sessionPreset = AVCaptureSession.Preset.hd1920x1080

        #if os(iOS)
        rotationCoordinator = AVCaptureDevice.RotationCoordinator(device: videoDevice, previewLayer: nil)
        rotationObservation = observe(\.rotationCoordinator!.videoRotationAngleForHorizonLevelCapture, options: [.initial, .new]) { [weak self] _, change in
            if let nv = change.newValue {
                self?.updateRotation(rotation: nv)
            }
        }
        #endif
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let currentTime = CFAbsoluteTimeGetCurrent()
        if currentTime - lastFrameTime < frameThrottleInterval {
            return // Skip this frame
        }
        lastFrameTime = currentTime
        
        if sampleBuffer.isValid && sampleBuffer.imageBuffer != nil {
            framesContinuation?.yield(sampleBuffer)
        }
    }
}
