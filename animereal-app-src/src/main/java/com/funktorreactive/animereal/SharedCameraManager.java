package com.funktorreactive.animereal;


import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.ConditionVariable;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.util.Size;
import android.view.Surface;

import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Session;
import com.google.ar.core.SharedCamera;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.UnavailableException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/*
Class for managing the state of the camera (as shared with ARCore)
 */
public class SharedCameraManager implements ImageReader.OnImageAvailableListener, SurfaceTexture.OnFrameAvailableListener {
    SharedCamera sharedCamera;
    Session sharedSession;
    MainActivity parent;

    // Reference to the camera system service.
    private CameraManager cameraManager;

    ImageReader cpuImageReader = null;

    // A list of CaptureRequest keys that can cause delays when switching between AR and non-AR modes.
    private List<CaptureRequest.Key<?>> keysThatCanCauseCaptureDelaysWhenModified;

    private CaptureRequest.Builder previewCaptureRequestBuilder;

    // Ensure GL surface draws only occur when new frames are available.
    private final AtomicBoolean shouldUpdateSurfaceTexture = new AtomicBoolean(false);

    // A check mechanism to ensure that the camera closed properly so that the app can safely exit.
    private final ConditionVariable safeToExitApp = new ConditionVariable();

    private HandlerThread backgroundThread;
    private Handler backgroundHandler;

    private boolean captureSessionChangesPossible = true;

    private boolean cameraPreviouslyOpened = false;

    private Image latestImage = null;

    CameraDevice cameraDevice;
    CameraCaptureSession captureSession;

    private boolean arcoreActive;


    // Camera ID for the camera used by ARCore.
    private String cameraId;

    private static final String TAG = "AnimeRealPoseCameraManager";

    private static final String HANDLE_THREAD_NAME = "AnimeRealCameraBackground";




    // Camera device state callback.
    private final CameraDevice.StateCallback cameraDeviceCallback =
            new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice cameraDevice) {
                    Log.d(TAG, "Camera device ID " + cameraDevice.getId() + " opened.");
                    SharedCameraManager.this.cameraDevice = cameraDevice;
                    createCameraPreviewSession();
                }

                @Override
                public void onClosed(CameraDevice cameraDevice) {
                    Log.d(TAG, "Camera device ID " + cameraDevice.getId() + " closed.");
                    SharedCameraManager.this.cameraDevice = null;
                    safeToExitApp.open();
                }

                @Override
                public void onDisconnected(CameraDevice cameraDevice) {
                    Log.w(TAG, "Camera device ID " + cameraDevice.getId() + " disconnected.");
                    cameraDevice.close();
                    SharedCameraManager.this.cameraDevice = null;
                }

                @Override
                public void onError(CameraDevice cameraDevice, int error) {
                    Log.e(TAG, "Camera device ID " + cameraDevice.getId() + " error " + error);
                    cameraDevice.close();
                    SharedCameraManager.this.cameraDevice = null;
                    // Fatal error. Quit application.
                    parent.finish();
                }
            };

    // Repeating camera capture session state callback.
    CameraCaptureSession.StateCallback cameraCaptureCallback =
            new CameraCaptureSession.StateCallback() {

                // Called when the camera capture session is first configured after the app
                // is initialized, and again each time the activity is resumed.
                @Override
                public void onConfigured(@NonNull CameraCaptureSession session) {
                    Log.d(TAG, "Camera capture session configured.");
                    captureSession = session;
                    setRepeatingCaptureRequest();
                    // Note, resumeARCore() must be called in onActive(), not here.
                }

                @Override
                public void onSurfacePrepared(
                        @NonNull CameraCaptureSession session, @NonNull Surface surface) {
                    Log.d(TAG, "Camera capture surface prepared.");
                }

                @Override
                public void onReady(@NonNull CameraCaptureSession session) {
                    Log.d(TAG, "Camera capture session ready.");
                }

                @Override
                public void onActive(@NonNull CameraCaptureSession session) {
                    Log.d(TAG, "Camera capture session active.");
                    if (!arcoreActive) {
                        resumeARCore();
                    }
                    synchronized (SharedCameraManager.this) {
                        captureSessionChangesPossible = true;
                        SharedCameraManager.this.notify();
                    }
                }

                @Override
                public void onCaptureQueueEmpty(@NonNull CameraCaptureSession session) {
                    Log.w(TAG, "Camera capture queue empty.");
                }

                @Override
                public void onClosed(@NonNull CameraCaptureSession session) {
                    Log.d(TAG, "Camera capture session closed.");
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                    Log.e(TAG, "Failed to configure camera capture session.");
                }
            };

    // Repeating camera capture session capture callback.
    private final CameraCaptureSession.CaptureCallback captureSessionCallback =
            new CameraCaptureSession.CaptureCallback() {

                @Override
                public void onCaptureCompleted(
                        @NonNull CameraCaptureSession session,
                        @NonNull CaptureRequest request,
                        @NonNull TotalCaptureResult result) {
                    shouldUpdateSurfaceTexture.set(true);
                }

                @Override
                public void onCaptureBufferLost(
                        @NonNull CameraCaptureSession session,
                        @NonNull CaptureRequest request,
                        @NonNull Surface target,
                        long frameNumber) {
                    Log.e(TAG, "onCaptureBufferLost: " + frameNumber);
                }

                @Override
                public void onCaptureFailed(
                        @NonNull CameraCaptureSession session,
                        @NonNull CaptureRequest request,
                        @NonNull CaptureFailure failure) {
                    Log.e(TAG, "onCaptureFailed: " + failure.getFrameNumber() + " " + failure.getReason());
                }

                @Override
                public void onCaptureSequenceAborted(
                        @NonNull CameraCaptureSession session, int sequenceId) {
                    Log.e(TAG, "onCaptureSequenceAborted: " + sequenceId + " " + session);
                }
            };

    public void onCreate() {
        resumeARCore();
    }

    // Surface texture on frame available callback, used only in non-AR mode.
    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        // Log.d(TAG, "onFrameAvailable()");
    }

    private synchronized void waitUntilCameraCaptureSessionIsActive() {
        while (!captureSessionChangesPossible) {
            try {
                this.wait();
            } catch (InterruptedException e) {
                Log.e(TAG, "Unable to wait for a safe time to make changes to the capture session", e);
            }
        }
    }


    public void onResume() {
        waitUntilCameraCaptureSessionIsActive();
        startBackgroundThread();
        if (cameraPreviouslyOpened) {
            openCamera();
        }

        //displayRotationHelper.onResume();
    }

    public void onPause() {
        waitUntilCameraCaptureSessionIsActive();
        //displayRotationHelper.onPause();
        pauseARCore();
        closeCamera();
        stopBackgroundThread();
    }


    private void resumeARCore() {
        // Ensure that session is valid before triggering ARCore resume. Handles the case where the user
        // manually uninstalls ARCore while the app is paused and then resumes.
        if (sharedSession == null) {
            return;
        }

        if (!arcoreActive) {
            /* try {*/
                // Resume ARCore.
                //sharedSession.resume();
                arcoreActive = true;

                // Set capture session callback while in AR mode.
                sharedCamera.setCaptureCallback(captureSessionCallback, backgroundHandler);
            /*} catch (CameraNotAvailableException e) {
                Log.e(TAG, "Failed to resume ARCore session", e);
                return;
            }*/
        }
    }

    private void pauseARCore() {
        shouldUpdateSurfaceTexture.set(false);
        if (arcoreActive) {
            sharedSession.pause();
            arcoreActive = false;
        }
    }

    // Called when starting non-AR mode or switching to non-AR mode.
    // Also called when app starts in AR mode, or resumes in AR mode.
    private void setRepeatingCaptureRequest() {
        try {
            setCameraEffects(previewCaptureRequestBuilder);

            captureSession.setRepeatingRequest(
                    previewCaptureRequestBuilder.build(), captureSessionCallback, backgroundHandler);
        } catch (CameraAccessException e) {
            Log.e(TAG, "Failed to set repeating request", e);
        }
    }

    // Close the camera device.
    private void closeCamera() {
        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }
        if (cameraDevice != null) {
            waitUntilCameraCaptureSessionIsActive();
            safeToExitApp.close();
            cameraDevice.close();
            safeToExitApp.block();
        }
        if (cpuImageReader != null) {
            cpuImageReader.close();
            cpuImageReader = null;
        }
    }

    private void createCameraPreviewSession() {
        try {
            // Note that isGlAttached will be set to true in AR mode in onDrawFrame().
            //sharedSession.setCameraTextureName(backgroundRenderer.getTextureId());
            sharedCamera.getSurfaceTexture().setOnFrameAvailableListener(this);

            // Create an ARCore compatible capture request using `TEMPLATE_RECORD`.
            previewCaptureRequestBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);

            // Build surfaces list, starting with ARCore provided surfaces.
            List<Surface> surfaceList = sharedCamera.getArCoreSurfaces();

            // Add a CPU image reader surface. On devices that don't support CPU image access, the image
            // may arrive significantly later, or not arrive at all.
            surfaceList.add(cpuImageReader.getSurface());

            // Surface list should now contain three surfaces:
            // 0. sharedCamera.getSurfaceTexture()
            // 1. …
            // 2. cpuImageReader.getSurface()

            // Add ARCore surfaces and CPU image surface targets.
            for (Surface surface : surfaceList) {
                previewCaptureRequestBuilder.addTarget(surface);
            }

            // Wrap our callback in a shared camera callback.
            CameraCaptureSession.StateCallback wrappedCallback =
                    sharedCamera.createARSessionStateCallback(cameraCaptureCallback, backgroundHandler);

            // Create camera capture session for camera preview using ARCore wrapped callback.
            cameraDevice.createCaptureSession(surfaceList, wrappedCallback, backgroundHandler);
        } catch (CameraAccessException e) {
            Log.e(TAG, "CameraAccessException", e);
        }
    }
    // Start background handler thread, used to run callbacks without blocking UI thread.
    private void startBackgroundThread() {
        backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    // Stop background handler thread.
    private void stopBackgroundThread() {
        if (backgroundThread != null) {
            backgroundThread.quitSafely();
            try {
                backgroundThread.join();
                backgroundThread = null;
                backgroundHandler = null;
            } catch (InterruptedException e) {
                Log.e(TAG, "Interrupted while trying to join background handler thread", e);
            }
        }
    }

    // Perform various checks, then open camera device and create CPU image reader.
    private void openCamera() {
        // Don't open camera if already opened.
        if (cameraDevice != null) {
            return;
        }

        if (sharedSession == null) {
            try {
                // Create ARCore session that supports camera sharing.
                sharedSession = new Session(this.parent, EnumSet.of(Session.Feature.SHARED_CAMERA));
            } catch (UnavailableException e) {
                Log.e(TAG, "Failed to create ARCore session that supports camera sharing", e);
                return;
            }

            // Enable auto focus mode while ARCore is running.
            Config config = sharedSession.getConfig();
            config.setFocusMode(Config.FocusMode.AUTO);
            sharedSession.configure(config);
        }

        // Store the ARCore shared camera reference.
        sharedCamera = sharedSession.getSharedCamera();

        // Store the ID of the camera used by ARCore.
        cameraId = sharedSession.getCameraConfig().getCameraId();

        // Use the currently configured CPU image size.
        Size desiredCpuImageSize = sharedSession.getCameraConfig().getImageSize();
        cpuImageReader =
                ImageReader.newInstance(
                        desiredCpuImageSize.getWidth(),
                        desiredCpuImageSize.getHeight(),
                        ImageFormat.YUV_420_888,
                        2);
        cpuImageReader.setOnImageAvailableListener(this, backgroundHandler);

        // When ARCore is running, make sure it also updates our CPU image surface.
        sharedCamera.setAppSurfaces(this.cameraId, Arrays.asList(cpuImageReader.getSurface()));

        try {

            // Wrap our callback in a shared camera callback.
            CameraDevice.StateCallback wrappedCallback =
                    sharedCamera.createARDeviceStateCallback(cameraDeviceCallback, backgroundHandler);

            // Store a reference to the camera system service.
            cameraManager = (CameraManager) this.parent.getSystemService(Context.CAMERA_SERVICE);

            // Get the characteristics for the ARCore camera.
            CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(this.cameraId);

            // On Android P and later, get list of keys that are difficult to apply per-frame and can
            // result in unexpected delays when modified during the capture session lifetime.
            if (Build.VERSION.SDK_INT >= 28) {
                keysThatCanCauseCaptureDelaysWhenModified = characteristics.getAvailableSessionKeys();
                if (keysThatCanCauseCaptureDelaysWhenModified == null) {
                    // Initialize the list to an empty list if getAvailableSessionKeys() returns null.
                    keysThatCanCauseCaptureDelaysWhenModified = new ArrayList<>();
                }
            }

            // Prevent app crashes due to quick operations on camera open / close by waiting for the
            // capture session's onActive() callback to be triggered.
            captureSessionChangesPossible = false;

            // Open the camera device using the ARCore wrapped callback.
            cameraManager.openCamera(cameraId, wrappedCallback, backgroundHandler);
        } catch (CameraAccessException | IllegalArgumentException | SecurityException e) {
            Log.e(TAG, "Failed to open camera", e);
        }
    }

    private <T> boolean checkIfKeyCanCauseDelay(CaptureRequest.Key<T> key) {
        if (Build.VERSION.SDK_INT >= 28) {
            // On Android P and later, return true if key is difficult to apply per-frame.
            return keysThatCanCauseCaptureDelaysWhenModified.contains(key);
        } else {
            // On earlier Android versions, log a warning since there is no API to determine whether
            // the key is difficult to apply per-frame. Certain keys such as CONTROL_AE_TARGET_FPS_RANGE
            // are known to cause a noticeable delay on certain devices.
            // If avoiding unexpected capture delays when switching between non-AR and AR modes is
            // important, verify the runtime behavior on each pre-Android P device on which the app will
            // be distributed. Note that this device-specific runtime behavior may change when the
            // device's operating system is updated.
            Log.w(
                    TAG,
                    "Changing "
                            + key
                            + " may cause a noticeable capture delay. Please verify actual runtime behavior on"
                            + " specific pre-Android P devices that this app will be distributed on.");
            // Allow the change since we're unable to determine whether it can cause unexpected delays.
            return false;
        }
    }

    // If possible, apply effect in non-AR mode, to help visually distinguish between from AR mode.
    private void setCameraEffects(CaptureRequest.Builder captureBuilder) {
        if (checkIfKeyCanCauseDelay(CaptureRequest.CONTROL_EFFECT_MODE)) {
            Log.w(TAG, "Not setting CONTROL_EFFECT_MODE since it can cause delays between transitions.");
        } else {
            Log.d(TAG, "Setting CONTROL_EFFECT_MODE to SEPIA in non-AR mode.");
            captureBuilder.set(
                    CaptureRequest.CONTROL_EFFECT_MODE, CaptureRequest.CONTROL_EFFECT_MODE_SEPIA);
        }
    }


    // CPU image reader callback.
    @Override
    public void onImageAvailable(ImageReader imageReader) {
        Image image = imageReader.acquireLatestImage();
        if (image == null) {
            Log.w(TAG, "onImageAvailable: Skipping null image.");
            return;
        }

        image.close();
    }

    public void openCameraFirstTime() {
        this.cameraPreviouslyOpened = true;
        openCamera();
    }


    public SharedCameraManager(MainActivity parent) {
        this.parent = parent;
        this.sharedSession = parent.getARSession();
    }

}
