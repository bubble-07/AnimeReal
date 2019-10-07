/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.funktorreactive.animereal;

import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.graphics.ImageFormat;
import android.graphics.Point;
import android.media.Image;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Display;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;
import com.google.ar.core.Anchor;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.Camera;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.HitTestResult;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.NodeParent;
import com.google.ar.sceneform.SkeletonNode;
import com.google.ar.sceneform.animation.ModelAnimator;
import com.google.ar.sceneform.collision.Ray;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.AnimationData;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.MaterialFactory;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.sceneform.ux.ArFragment;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Demonstrates playing animated FBX models. */
public class MainActivity extends AppCompatActivity {

  private static final String TAG = "AnimeReal";
  /*
  Renderable id corresponding to the anime character we want to render
   */
  private static final int CHARACTER_RENDERABLE = 1;

  private SkeletonNode character;

  private ModelRenderable characterRenderable;

  private BasicCylinderModel cylinderModel = new BasicCylinderModel();

  private BasicCylinderModel.DebugRenderer debugRenderer = null;

  private StandardBody standardBody = null;

  private StandardBodyCorrespondences standardBodyCorrespondences = null;


  private PoseAnnotatorManager annotatorManager;


  //TODO: Remove the following once you're done ripping this apart


  // The UI to play next animation.
  private FloatingActionButton animationButton;
  // The UI to toggle wearing the hat.
  private FloatingActionButton hatButton;

  //TODO: End "to delete" block



  private ArFragment arFragment;
  // Model loader class to avoid leaking the activity context.
  private ModelLoader modelLoader;

  private AnchorNode anchorNode;



  @Override
  @SuppressWarnings({"AndroidApiChecker", "FutureReturnValueIgnored"})
  protected void onCreate(Bundle savedInstanceState) {

    standardBody = new StandardBody(this);

    standardBodyCorrespondences = StandardBodyCorrespondences.getStandardCorrespondences(standardBody);

    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    arFragment = (ArFragment) getSupportFragmentManager().findFragmentById(R.id.sceneform_fragment);


    modelLoader = new ModelLoader(this);

    modelLoader.loadModel(CHARACTER_RENDERABLE, R.raw.character);

    // When a plane is tapped, the model is placed on an Anchor node anchored to the plane.
    arFragment.setOnTapArPlaneListener(this::onPlaneTap);

    // Add a frame update listener to the scene to control the state of the buttons.
    arFragment.getArSceneView().getScene().addOnUpdateListener(this::onFrameUpdate);

    arFragment.getArSceneView().getScene().addOnPeekTouchListener(this::onPeekTouch);

    // Once the model is placed on a plane, this button plays the animations.
    this.animationButton = findViewById(R.id.animate);
    animationButton.setEnabled(false);
    animationButton.setOnClickListener(this::onPlayAnimation);

    // Place or remove a hat on Andy's head showing how to use Skeleton Nodes.
    hatButton = findViewById(R.id.hat);
    hatButton.setEnabled(false);
    hatButton.setOnClickListener(this::onToggleHat);


    this.annotatorManager = new PoseAnnotatorManager(this);
    this.annotatorManager.startBackgroundThread();
  }

  public Session getARSession() {
    return this.arFragment.getArSceneView().getSession();
  }

  private void onPlayAnimation(View unusedView) {
    //DO NOTHING! We don't care right nao
  }

  private void onPeekTouch(HitTestResult hitTestResult, MotionEvent motionEvent) {
    if (this.character == null || characterRenderable == null) {
      return;
    }

    NodeParent ancestor = arFragment.getArSceneView().getScene();
    Material m = this.characterRenderable.getMaterial();
    this.debugRenderer = this.cylinderModel.getDebugRenderer(ancestor, m);

  }

  /*
   * Used as the listener for setOnTapArPlaneListener.
   */
  private void onPlaneTap(HitResult hitResult, Plane unusedPlane, MotionEvent unusedMotionEvent) {
    if (characterRenderable == null) {
      return;
    }
    // Create the Anchor.
    Anchor anchor = hitResult.createAnchor();

    if (anchorNode == null) {
      anchorNode = new AnchorNode(anchor);
      NodeParent ancestor = arFragment.getArSceneView().getScene();
      anchorNode.setParent(ancestor);

      character = new SkeletonNode();

      character.setParent(ancestor);
      character.setRenderable(characterRenderable);

      ArmatureUtils.rigVRChatArmatureNodes(character, characterRenderable);
      this.cylinderModel.adjustDimensionsToVRChatModel(character);
      ArmatureUtils.zeroRotationTree(character, characterRenderable);

    }
  }

  /**
   * Called on every frame, control the state of the buttons.
   *
   * @param frameTime
   */
  private void onFrameUpdate(FrameTime frameTime) {


    Frame arFrame = this.arFragment.getArSceneView().getArFrame();
    if (arFrame != null) {
      try {
        Image cameraImage = arFrame.acquireCameraImage();


        if (cameraImage != null) {

          Bitmap bitmapImage = ImageUtils.imageToBitmap(cameraImage);


          if (bitmapImage != null) {
            Log.e(TAG, "Bitmap available!");

            Bitmap rescaled = Bitmap.createScaledBitmap(bitmapImage, 256, 256, false);

            Bitmap rescaledCopy = rescaled.copy(Bitmap.Config.ARGB_8888, false);

            rescaled.recycle();
            bitmapImage.recycle();

            Ray[] cornerRays = getCornerRays();

            ObservationFrame observationFrame = new ObservationFrame(new TimeStamp(frameTime), cornerRays);

            AnnotatorInput input = new AnnotatorInput(observationFrame, rescaledCopy);

            this.annotatorManager.setCurrentAnnotationInput(input);
          }

          cameraImage.close();
        }
      }
      catch (NotYetAvailableException e) {
        Log.e(TAG, "Camera image not yet available", e);
      }
    }

    // If the model has not been placed yet, disable the buttons.
    if (anchorNode == null) {
      if (animationButton.isEnabled()) {
        animationButton.setBackgroundTintList(ColorStateList.valueOf(android.graphics.Color.GRAY));
        animationButton.setEnabled(false);
        hatButton.setBackgroundTintList(ColorStateList.valueOf(android.graphics.Color.GRAY));
        hatButton.setEnabled(false);
      }
    } else {
      if (!animationButton.isEnabled()) {
        animationButton.setBackgroundTintList(
            ColorStateList.valueOf(ContextCompat.getColor(this, R.color.colorAccent)));
        animationButton.setEnabled(true);
        hatButton.setEnabled(true);
        hatButton.setBackgroundTintList(
            ColorStateList.valueOf(ContextCompat.getColor(this, R.color.colorPrimary)));
      }
    }

    if (debugRenderer != null) {
      this.debugRenderer.adjustPosition();
    }

    //Update logic for displaying animu girl

    if (anchorNode != null) {

      //Dumb thing to just choose most recent output for now
      AnnotatorOutput chosenOutput = null;


      List<AnnotatorOutput> outputsList = this.annotatorManager.getAnnotatorOutputs();
      Log.e(TAG, "Outputs len " + Integer.toString(outputsList.size()));
      if (outputsList.size() >= 1) {
        chosenOutput = outputsList.get(outputsList.size() - 1);

        Log.e(TAG, "Start time: " + Float.toString(chosenOutput.getTime().getStartSeconds()));

        BodyNet<List<AnnotatorOutput.SingleObservation>> annotatorOutput = chosenOutput.getBodyNet(standardBody, 0.3f);

        annotatorOutput = annotatorOutput.transform(BodyNetMath::reduceAverage);

        List<BasicCylinderModel.RayCorrespondence> correspondences = this.standardBodyCorrespondences.getRayCorrespondences(annotatorOutput);

        float alpha = 0.001f; //0.01f;
        int num_steps = 4;

        this.cylinderModel.iterativeRayFit(num_steps, alpha, correspondences);

      }

      /*
      List<Ray> targets = new ArrayList<>();
      List<BasicCylinderModel.ModelLocation> modelLocations = new ArrayList<>();

      targets.add(getStandardizedRayThrough(500.0f, 500.0f));
      targets.add(getStandardizedRayThrough(600.0f, 600.0f));


      targets.add(getStandardizedRayThrough(0.0f, 0.0f));
      targets.add(getStandardizedRayThrough(1000.0f, 0.0f));
      targets.add(getStandardizedRayThrough(1000.0f, 1000.0f));
      targets.add(getStandardizedRayThrough(0.0f, 1000.0f));

      modelLocations.add(new BasicCylinderModel.ModelLocation(BasicCylinderModel.CylinderSegment.BODY, 0.0f, 1.0f));

      modelLocations.add(new BasicCylinderModel.ModelLocation(BasicCylinderModel.CylinderSegment.BODY, 1.5f, 0.0f));


      modelLocations.add(new BasicCylinderModel.ModelLocation(BasicCylinderModel.CylinderSegment.RIGHT_FOREARM, 0.0f, 1.0f));
      modelLocations.add(new BasicCylinderModel.ModelLocation(BasicCylinderModel.CylinderSegment.LEFT_FOREARM, 0.0f, 1.0f));

      modelLocations.add(new BasicCylinderModel.ModelLocation(BasicCylinderModel.CylinderSegment.LEFT_FORELEG, 0.0f, 1.0f));

      modelLocations.add(new BasicCylinderModel.ModelLocation(BasicCylinderModel.CylinderSegment.RIGHT_FORELEG, 0.0f, 1.0f));


      float alpha = 0.01f;
      int num_steps = 10;

      //Update cylinder node according to targets
      cylinderModel.iterativeRayFit(num_steps, alpha, modelLocations, targets);
      */

      cylinderModel.applyToVRChatModel(character);
    }
  }

  private Ray[] getCornerRays() {
    Display disp = this.getWindowManager().getDefaultDisplay();
    Point size = new Point();
    disp.getSize(size);
    int width = size.x;
    int height = size.y;

    return new Ray[]{getStandardizedRayThrough(0, 0), getStandardizedRayThrough(width, 0),
                     getStandardizedRayThrough(width, height), getStandardizedRayThrough(0, height)};
  }

  private Ray getStandardizedRayThrough(float x, float y) {
    //First, find the ray starting from the near clip plane
    Ray simpleRay = this.arFragment.getArSceneView().getScene().getCamera().screenPointToRay(x, y);

    //Get the camera's world-origin
    Vector3 cameraOrigin = this.arFragment.getArSceneView().getScene().getCamera().getWorldPosition();

    //Start the ray from there, going in the same direction
    return new Ray(cameraOrigin, simpleRay.getDirection());
  }

  private static float randFloat(float min, float max) {
    Random rand = new Random();

    float result = rand.nextFloat() * (max - min) + min;

    return result;
  }

  private void onToggleHat(View unused) {
    //DO NOTHING
  }

  void setRenderable(int id, ModelRenderable renderable) {
    if (id == this.CHARACTER_RENDERABLE) {
      this.characterRenderable = renderable;
    }
  }

  void onException(int id, Throwable throwable) {
    Toast toast = Toast.makeText(this, "Unable to load renderable: " + id, Toast.LENGTH_LONG);
    toast.setGravity(Gravity.CENTER, 0, 0);
    toast.show();
    Log.e(TAG, "Unable to load andy renderable", throwable);
  }

  public void onDestroy() {
    super.onDestroy();
      this.annotatorManager.onDestroy();
  }

  @Override
  public void onPause() {
    Log.e(TAG, "App closing");
    this.annotatorManager.stopBackgroundThread();
    super.onPause();
  }
  @Override
  public void onResume() {
    super.onResume();
    this.annotatorManager.startBackgroundThread();
  }
}
