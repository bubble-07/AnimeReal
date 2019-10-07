package com.funktorreactive.animereal;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/*
Class responsible for managing thread state corresponding to the pose annotator(s)
running in the background
 */
public class PoseAnnotatorManager {

    private static final String TAG = "AnimeRealPoseAnnotatorManager";

    private int NUM_OUTPUTS_TO_KEEP = 1;

    private AnnotatorOutput currentOutput = null;


    private final Object lock = new Object();
    private boolean runannotator = false;
    private PoseAnnotator annotator;
    boolean currentUseNNAPI = false;
    int currentNumThreads = 4;

    private HandlerThread backgroundThread;
    private Handler backgroundHandler;

    private static final String HANDLE_THREAD_NAME = "AnimeRealBackgroundThread";

    private Activity parent;

    public PoseAnnotatorManager(Activity parent) {
        this.parent = parent;
    }

    private synchronized void addAnnotatorOutput(AnnotatorOutput output) {
        this.currentOutput = output;
    }

    public synchronized List<AnnotatorOutput> getAnnotatorOutputs() {
        ArrayList<AnnotatorOutput> result = new ArrayList<>();
        result.add(this.currentOutput);
        return result;
    }

    /** Starts a background thread and its {@link Handler}. */
    public void startBackgroundThread() {
        backgroundThread = new HandlerThread(HANDLE_THREAD_NAME);
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
        // Start the classification train & load an initial model.
        synchronized (lock) {
            runannotator = true;
        }
        backgroundHandler.post(periodicClassify);
        updateActiveModel();
    }

    /** Stops the background thread and its {@link Handler}. */
    public void stopBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
            synchronized (lock) {
                runannotator = false;
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted when stopping background thread", e);
        }
    }

    /** Takes photos and classify them periodically. */
    private Runnable periodicClassify =
            new Runnable() {
                @Override
                public void run() {
                    synchronized (lock) {
                        if (runannotator) {
                            classifyFrame();
                        }
                    }
                    backgroundHandler.post(periodicClassify);
                }
            };

    /** Classifies a frame from the preview stream. */
    private void classifyFrame() {
        if (annotator == null) {
            return;
        }
        AnnotatorInput input = this.getCurrentAnnotationInput();
        if (input == null) {
            return;
        }
        else if (input.getBitmap() == null) {
            return;
        }
        AnnotatorOutput output = annotator.annotateFrame(input);
        this.addAnnotatorOutput(output);
    }

    private void updateActiveModel() {

        backgroundHandler.post(() -> {

            // Disable annotator while updating
            if (annotator != null) {
                annotator.close();
                annotator = null;
            }

            // Try to load model.
            try {
                annotator = new PoseAnnotatorSlimQuantized(this.parent);
            } catch (IOException e) {
                Log.d(TAG, "Failed to load", e);
                annotator = null;
            }

            // Customize the interpreter to the type of device we want to use.
            if (annotator == null) {
                return;
            }
            annotator.setNumThreads(this.currentNumThreads);
            if (this.currentUseNNAPI) {
                annotator.useNNAPI();
            }
        });
    }

    AnnotatorInput currentAnnotationInput = null;

    public synchronized void setCurrentAnnotationInput(AnnotatorInput input) {
        AnnotatorInput oldInput = this.getCurrentAnnotationInput();
        this.currentAnnotationInput = input;
        if (oldInput != null) {
            oldInput.recycle();
        }
    }

    public synchronized AnnotatorInput getCurrentAnnotationInput() {
        return this.currentAnnotationInput;
    }

    public void onDestroy() {
        if (annotator != null) {
            annotator.close();
        }
    }
}
