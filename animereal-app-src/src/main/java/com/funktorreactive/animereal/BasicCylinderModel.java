package com.funktorreactive.animereal;

import android.graphics.ColorSpace;
import android.util.Log;

import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.NodeParent;
import com.google.ar.sceneform.SkeletonNode;
import com.google.ar.sceneform.collision.Ray;
import com.google.ar.sceneform.math.Quaternion;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Material;
import com.google.ar.sceneform.rendering.Renderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.google.ar.schemas.lull.Quat;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import java.util.function.Function;

/*
Represents the "basic cylinder model" of the human body, which consists of
fore (arm) / (arm) cylinders attached to a cylindrical body, with a cylindrical head,
with the coordinate base at the ~genitals~
 */
public class BasicCylinderModel {
    private static final String TAG = "AnimeReal";

    private static CylinderSegment[] topoSortedSegments = {CylinderSegment.BODY,
    CylinderSegment.HEAD, CylinderSegment.RIGHT_LEG, CylinderSegment.LEFT_LEG,
    CylinderSegment.RIGHT_ARM, CylinderSegment.LEFT_ARM,
    CylinderSegment.RIGHT_FORELEG, CylinderSegment.LEFT_FORELEG,
    CylinderSegment.RIGHT_FOREARM, CylinderSegment.LEFT_FOREARM};


    public enum CylinderSegment {
        LEFT_ARM(0, "LeftArm"), LEFT_FOREARM(1, "LeftForeArm"),
        RIGHT_ARM(2, "RightArm"), RIGHT_FOREARM(3, "RightForeArm"),
        LEFT_LEG(4, "LeftUpLeg"), LEFT_FORELEG(5, "LeftLeg"),
        RIGHT_LEG(6, "RightUpLeg"), RIGHT_FORELEG(7, "RightLeg"),
        BODY(8, "Hips"), HEAD(9, "Neck");

        static  {
            LEFT_ARM.parent = BODY;
            RIGHT_ARM.parent = BODY;
            LEFT_LEG.parent = BODY;
            RIGHT_LEG.parent = BODY;
            HEAD.parent = BODY;

            LEFT_FOREARM.parent = LEFT_ARM;
            RIGHT_FOREARM.parent = RIGHT_ARM;
            LEFT_FORELEG.parent = LEFT_LEG;
            RIGHT_FORELEG.parent = RIGHT_LEG;

            BODY.parent = null;


            LEFT_FOREARM.measuredVRChatEndBone = "LeftHandMiddle3";
            RIGHT_FOREARM.measuredVRChatEndBone = "RightHandMiddle3";
            LEFT_FORELEG.measuredVRChatEndBone = "LeftFoot";
            RIGHT_FORELEG.measuredVRChatEndBone = "RightFoot";

            HEAD.measuredVRChatEndBone = "HeadTop_End";
            BODY.measuredVRChatEndBone = "Neck";
            LEFT_ARM.measuredVRChatEndBone = "LeftForeArm";
            RIGHT_ARM.measuredVRChatEndBone = "RightForeArm";
            LEFT_LEG.measuredVRChatEndBone = "LeftLeg";
            RIGHT_LEG.measuredVRChatEndBone = "RightLeg";
        }

        private final int index;
        private final String controlledVRChatBone;
        private CylinderSegment parent;
        private String measuredVRChatEndBone;
        CylinderSegment(int index, String vrChatName) {
            this.index = index;
            this.controlledVRChatBone = vrChatName;
        }
        public int getIndex() {
            return this.index;
        }
        public String getVRChatBoneName() {
            return this.controlledVRChatBone;
        }

        public String getVRChatEndBoneName() {
            return this.measuredVRChatEndBone;
        }

        public CylinderSegment getParent() {
            return this.parent;
        }
    }

    private float[] cylinderLengths = new float[10];
    private float[] cylinderRadii = new float[10];

    //Total rotation of components is determined by first rotating by the
    //initial quaternions (which yields the transformation _into_ the T-pose)
    //and then rotating by the cylinder quaternions (from T-pose to current pose)
    private List<Quaternion> cylinderQuaternions = new ArrayList<>();
    private List<Quaternion> initialQuaternions = new ArrayList<>();
    private Vector3 initialTranslation = new Vector3();

    public BasicCylinderModel() {

        float inToMetersFactor = 25.40f / 1000.0f;

        //Initialization values here are based on Alex Grabanski's approx measurements

        cylinderLengths[CylinderSegment.LEFT_ARM.getIndex()] = 13.0f;
        cylinderLengths[CylinderSegment.RIGHT_ARM.getIndex()] = 13.0f;

        cylinderLengths[CylinderSegment.LEFT_FOREARM.getIndex()] = 16.0f;
        cylinderLengths[CylinderSegment.RIGHT_FOREARM.getIndex()] = 16.0f;

        cylinderLengths[CylinderSegment.LEFT_LEG.getIndex()] = 16.0f;
        cylinderLengths[CylinderSegment.RIGHT_LEG.getIndex()] = 16.0f;

        cylinderLengths[CylinderSegment.LEFT_FORELEG.getIndex()] = 18.0f;
        cylinderLengths[CylinderSegment.RIGHT_FORELEG.getIndex()] = 18.0f;

        cylinderLengths[CylinderSegment.BODY.getIndex()] = 17.0f;

        cylinderLengths[CylinderSegment.HEAD.getIndex()] = 12.0f;

        //Convert lengths from inches to meters
        for (int i = 0; i < cylinderLengths.length; i++) {
            cylinderLengths[i] = cylinderLengths[i] * inToMetersFactor;
        }


        cylinderRadii[CylinderSegment.LEFT_ARM.getIndex()] = 4.0f;
        cylinderRadii[CylinderSegment.RIGHT_ARM.getIndex()] = 4.0f;

        cylinderRadii[CylinderSegment.LEFT_FOREARM.getIndex()] = 3.0f;
        cylinderRadii[CylinderSegment.RIGHT_FOREARM.getIndex()] = 3.0f;

        cylinderRadii[CylinderSegment.LEFT_LEG.getIndex()] = 6.0f;
        cylinderRadii[CylinderSegment.RIGHT_LEG.getIndex()] = 6.0f;

        cylinderRadii[CylinderSegment.LEFT_FORELEG.getIndex()] = 5.0f;
        cylinderRadii[CylinderSegment.RIGHT_FORELEG.getIndex()] = 5.0f;

        cylinderRadii[CylinderSegment.BODY.getIndex()] = 14.0f;

        cylinderRadii[CylinderSegment.HEAD.getIndex()] = 8.0f;

        //Convert diameters in inches to radii in meters
        for (int i = 0; i < cylinderRadii.length; i++) {
            cylinderRadii[i] = cylinderRadii[i] * (inToMetersFactor / 2.0f);
        }

        //Fill quaternion values
        for (int i = 0; i < cylinderRadii.length; i++) {
            cylinderQuaternions.add(new Quaternion());
            initialQuaternions.add(new Quaternion());
        }

        Vector3 intoScreen = new Vector3(0.0f, 1.0f, 0.0f);

        Vector3 horiz = new Vector3(1.0f, 0.0f, 0.0f);

        Quaternion armDeclination = Quaternion.axisAngle(horiz, 180.0f);


        Quaternion defaultViewQuaternion = Quaternion.axisAngle(Vector3.right(), -90.0f);

        initialQuaternions.set(CylinderSegment.BODY.getIndex(), defaultViewQuaternion);



        //Set the model up in a T-pose with inverted legs
        Quaternion leftArmRot = Quaternion.axisAngle(intoScreen, 90.0f);
        leftArmRot = Quaternion.multiply(armDeclination, leftArmRot);
        cylinderQuaternions.set(CylinderSegment.LEFT_ARM.getIndex(), leftArmRot);

        Quaternion rightArmRot = Quaternion.axisAngle(intoScreen, -90.0f);
        rightArmRot = Quaternion.multiply(armDeclination, rightArmRot);
        cylinderQuaternions.set(CylinderSegment.RIGHT_ARM.getIndex(), rightArmRot);



        //Set the model up in a T-pose
        Quaternion leftLegRot = Quaternion.axisAngle(intoScreen, 180.0f);
        cylinderQuaternions.set(CylinderSegment.LEFT_LEG.getIndex(), leftLegRot);

        Quaternion rightLegRot = Quaternion.axisAngle(intoScreen, -180.0f);
        rightLegRot = Quaternion.multiply(Quaternion.axisAngle(new Vector3(0.0f, 0.0f, 1.0f), 180.0f), rightLegRot);
        cylinderQuaternions.set(CylinderSegment.RIGHT_LEG.getIndex(), rightLegRot);


    }

    public DebugRenderer getDebugRenderer(NodeParent ancestor, Material m) {
        return new DebugRenderer(ancestor, m);
    }

    /*
    Object for rendering a BasicCylinderModel as cylinders
     */
    public class DebugRenderer {
        List<Node> nodes = new ArrayList<>();
        List<Renderable> renderables = new ArrayList<>();

        public DebugRenderer(NodeParent ancestor, Material m) {
            for (CylinderSegment segment : CylinderSegment.values()) {
                nodes.add(null);
                renderables.add(null);
            }

            for (CylinderSegment segment : topoSortedSegments) {
                Node segmentNode = new Node();

                if (segment.getParent() == null) {
                    segmentNode.setParent(ancestor);
                }
                else {
                    CylinderSegment parentSeg = segment.getParent();

                    Node parent = nodes.get(parentSeg.getIndex());
                    segmentNode.setParent(parent);

                    Vector3 attachPoint = getAttachPoint(parentSeg, segment);
                    segmentNode.setLocalPosition(attachPoint);
                }

                Node cylinderNode = new Node();
                cylinderNode.setParent(segmentNode);

                float radius = cylinderRadii[segment.getIndex()];
                float height = cylinderLengths[segment.getIndex()];
                Vector3 center = new Vector3(0.0f, 0.0f, 0.0f);

                Renderable segmentRenderable = ShapeFactory.makeCylinder(radius, height, center, m);

                Quaternion cylinderFix = Quaternion.axisAngle(new Vector3(1.0f, 0.0f, 0.0f), 90.0f);
                Vector3 cylinderTranslationFix = new Vector3(0.0f, 0.0f, height / 2.0f);

                cylinderNode.setLocalPosition(cylinderTranslationFix);
                cylinderNode.setLocalRotation(cylinderFix);
                cylinderNode.setRenderable(segmentRenderable);

                this.nodes.set(segment.getIndex(), segmentNode);
                this.renderables.set(segment.getIndex(), segmentRenderable);
            }
            adjustPosition();
        }
        public void adjustPosition() {
            for (CylinderSegment segment : CylinderSegment.values()) {
                int ind = segment.getIndex();

                Quaternion q = getSegmentTotalRotation(segment);
                Node n = this.nodes.get(ind);
                if (segment == CylinderSegment.BODY) {
                    n.setWorldRotation(q);
                }
                else {
                    n.setLocalRotation(q);
                }
            }
            Node body = this.nodes.get(CylinderSegment.BODY.getIndex());
            if (body != null) {
                body.setWorldPosition(initialTranslation);
            }
        }
    }

    private Quaternion getSegmentTotalRotation(CylinderSegment segment) {
        int ind = segment.getIndex();
        Quaternion cylinderQ = cylinderQuaternions.get(ind);
        Quaternion initialQ = initialQuaternions.get(ind);
        return Quaternion.multiply(cylinderQ, initialQ);
    }

    public void scaleDimensions(float scaleFac) {
        for (CylinderSegment segment : CylinderSegment.values()) {
            int ind = segment.getIndex();
            this.cylinderLengths[ind] *= scaleFac;
            this.cylinderRadii[ind] *= scaleFac;
        }
    }

    /*
    Given a SkeletonNode, determine the empirical lengths of cylinders, and use
    the current ratios with radii to set the new radii
     */
    public void adjustDimensionsToVRChatModel(SkeletonNode character) {
        for (CylinderSegment segment : CylinderSegment.values()) {
            Node boneNode = character.getBoneAttachment(segment.getVRChatBoneName());
            Node endBoneNode = character.getBoneAttachment(segment.getVRChatEndBoneName());
            if (boneNode != null && endBoneNode != null) {
                Vector3 one = boneNode.getWorldPosition();
                Vector3 two = endBoneNode.getWorldPosition();
                Vector3 diff = Vector3.subtract(one, two);
                float sqDist = Vector3.dot(diff, diff);
                float dist = (float) Math.sqrt((double) sqDist);


                int ind = segment.getIndex();
                //Okay, great, perform the adjustment
                float oldRadiusRatio = this.cylinderRadii[ind] / this.cylinderLengths[ind];

                this.cylinderLengths[ind] = dist;
                this.cylinderRadii[ind] = dist * oldRadiusRatio;
            }
        }
    }

    public void applyToVRChatModel(SkeletonNode character) {
        //First, set all of the rotation quaternions correctly
        for (CylinderSegment segment : CylinderSegment.values()) {
            Node boneNode = character.getBoneAttachment(segment.getVRChatBoneName());
            if (boneNode != null) {
                int segmentInd = segment.getIndex();
                //Quaternion for the bone in the cylinder model, as adjusted from the t-pose
                Quaternion currentModelQuaternion = this.cylinderQuaternions.get(segmentInd);
                //Note: it's assumed that
                //the T-pose involves all identity rotations everywhere for the display node
                //that we're setting the local rotation of, except for the left up leg and right up leg
                Quaternion defaultViewQuaternion = new Quaternion();

                /*
                if (segment == CylinderSegment.LEFT_LEG || segment == CylinderSegment.RIGHT_LEG) {
                    defaultViewQuaternion = new Quaternion(0.0f, 0.0f, 1.0f, 0.0f);
                }
                if (segment == CylinderSegment.BODY) {
                    defaultViewQuaternion = new Quaternion(0.0f, 1.0f, 0.0f, 0.0f);
                    defaultViewQuaternion = Quaternion.multiply(Quaternion.axisAngle(Vector3.right(), 90.0f), defaultViewQuaternion);
                    defaultViewQuaternion = Quaternion.multiply(Quaternion.axisAngle(Vector3.forward(), 180.0f), defaultViewQuaternion);
                }
                if (segment == CylinderSegment.RIGHT_ARM) {
                    defaultViewQuaternion = (new Quaternion(-0.452f, 0.616f, -0.51f, -0.38f)).inverted();
                }
                if (segment == CylinderSegment.LEFT_ARM) {
                    defaultViewQuaternion = (new Quaternion(0.452f, 0.616f, -0.51f, 0.389f)).inverted();
                }*/

                //uaternion inverseDefaultView = defaultViewQuaternion.inverted();


                Quaternion resultQuaternion = Quaternion.multiply(currentModelQuaternion, defaultViewQuaternion);

                if (segment == CylinderSegment.BODY) {
                    boneNode.setWorldRotation(resultQuaternion);
                }
                else {
                    boneNode.setLocalRotation(resultQuaternion);
                }

            }
        }
        //Set translation of whole model correctly
        Node hipsNode = character.getBoneAttachment(CylinderSegment.BODY.getVRChatBoneName());
        if (hipsNode != null) {
            hipsNode.setWorldPosition(this.initialTranslation);
        }

    }

    private Vector3 cylindrical(float r, float theta, float z) {
        float cos = (float) Math.cos(theta);
        float sin = (float) Math.sin(theta);

        return new Vector3(r * cos, r * sin, z);
    }


    public Vector3 getAttachPoint(CylinderSegment parent, CylinderSegment child) {
        float body_height = cylinderLengths[CylinderSegment.BODY.getIndex()];
        float body_radius = cylinderRadii[CylinderSegment.BODY.getIndex()];

        if (parent == CylinderSegment.BODY && child == CylinderSegment.RIGHT_ARM) {
            float arm_radius = cylinderRadii[CylinderSegment.RIGHT_ARM.getIndex()];
            return new Vector3(-body_radius, 0.0f, body_height - arm_radius);
        }

        if (parent == CylinderSegment.BODY && child == CylinderSegment.LEFT_ARM) {
            float arm_radius = cylinderRadii[CylinderSegment.LEFT_ARM.getIndex()];
            return new Vector3(body_radius, 0.0f, body_height - arm_radius);
        }

        if (parent == CylinderSegment.BODY && child == CylinderSegment.LEFT_LEG) {
            float leg_radius = cylinderRadii[CylinderSegment.LEFT_LEG.getIndex()];
            return new Vector3(body_radius - leg_radius, 0.0f, 0.0f);
        }

        if (parent == CylinderSegment.BODY && child == CylinderSegment.RIGHT_LEG) {
            float leg_radius = cylinderRadii[CylinderSegment.RIGHT_LEG.getIndex()];
            return new Vector3(-body_radius + leg_radius, 0.0f, 0.0f);
        }

        if (parent == CylinderSegment.BODY && child == CylinderSegment.HEAD) {
            return new Vector3(0.0f, 0.0f, body_height);
        }

        if (parent == CylinderSegment.RIGHT_ARM && child == CylinderSegment.RIGHT_FOREARM) {
            float arm_length = cylinderLengths[CylinderSegment.RIGHT_ARM.getIndex()];
            return new Vector3(0.0f, 0.0f, arm_length);
        }

        if (parent == CylinderSegment.LEFT_ARM && child == CylinderSegment.LEFT_FOREARM) {
            float arm_length = cylinderLengths[CylinderSegment.LEFT_ARM.getIndex()];
            return new Vector3(0.0f, 0.0f, arm_length);
        }

        if (parent == CylinderSegment.LEFT_LEG && child == CylinderSegment.LEFT_FORELEG) {
            float leg_length = cylinderLengths[CylinderSegment.LEFT_LEG.getIndex()];
            return new Vector3(0.0f, 0.0f, leg_length);
        }

        if (parent == CylinderSegment.RIGHT_LEG && child == CylinderSegment.RIGHT_FORELEG) {
            float leg_length = cylinderLengths[CylinderSegment.RIGHT_LEG.getIndex()];
            return new Vector3(0.0f, 0.0f, leg_length);
        }
        return null;
    }

    /*
    Gets the (post-quaternion-rotation) local point for a given segment and [normalized] cylindrical coord
     */
    public Vector3 getRotLocalPointAt(CylinderSegment segment, float theta, float z) {
        int ind = segment.getIndex();
        float cylinderLen = cylinderLengths[ind];
        Vector3 local = cylindrical(cylinderRadii[ind], theta, cylinderLen * z);
        return Quaternion.rotateVector(cylinderQuaternions.get(ind), local);
    }

    /*
    converts rot-local coordinates in child space to rot-local parent coords
     */
    public Vector3 convertLocal(CylinderSegment parent, CylinderSegment child, Vector3 childRotLocal) {
        Vector3 attachPoint = getAttachPoint(parent, child);
        return Quaternion.rotateVector(this.getSegmentTotalRotation(parent), Vector3.add(attachPoint, childRotLocal));
    }


    /*
    Gets the root-relative (same rotation frame as world coords, but without any translation from "genitals = 0")
     point corresponding to a position on the given cylinder segment
    given by local cylindrical coordinates
     */
    public Vector3 getRootRelativePointAt(CylinderSegment segment, float theta, float z) {
        Vector3 local = getRotLocalPointAt(segment, theta, z);

        switch (segment) {
            case LEFT_FOREARM:
                segment = CylinderSegment.LEFT_ARM;
                local = convertLocal(CylinderSegment.LEFT_ARM, CylinderSegment.LEFT_FOREARM, local);
                break;
            case RIGHT_FOREARM:
                segment = CylinderSegment.RIGHT_ARM;
                local = convertLocal(CylinderSegment.RIGHT_ARM, CylinderSegment.RIGHT_FOREARM, local);
                break;
            case LEFT_FORELEG:
                segment = CylinderSegment.LEFT_LEG;
                local = convertLocal(CylinderSegment.LEFT_LEG, CylinderSegment.LEFT_FORELEG, local);
                break;
            case RIGHT_FORELEG:
                segment = CylinderSegment.RIGHT_LEG;
                local = convertLocal(CylinderSegment.RIGHT_LEG, CylinderSegment.RIGHT_FORELEG, local);
                break;
        }

        switch (segment) {
            case BODY:
                return local;
            case RIGHT_ARM:
                return convertLocal(CylinderSegment.BODY, CylinderSegment.RIGHT_ARM, local);
            case LEFT_ARM:
                return convertLocal(CylinderSegment.BODY, CylinderSegment.LEFT_ARM, local);
            case LEFT_LEG:
                return convertLocal(CylinderSegment.BODY, CylinderSegment.LEFT_LEG, local);
            case RIGHT_LEG:
                return convertLocal(CylinderSegment.BODY, CylinderSegment.RIGHT_LEG, local);
            case HEAD:
                return convertLocal(CylinderSegment.BODY, CylinderSegment.HEAD, local);

        }
        return null;
    }

    public Vector3 getGlobalPointAt(CylinderSegment segment, float theta, float z) {
        return Vector3.add(this.initialTranslation, this.getRootRelativePointAt(segment, theta, z));
    }

    public Vector3 getGlobalPointAt(ModelLocation location) {
        return this.getGlobalPointAt(location.getSegment(), location.getTheta(), location.getZ());
    }

    public static class ModelLocation {
        CylinderSegment segment;
        float theta;
        float z;
        public ModelLocation(CylinderSegment segment, float theta, float z) {
            this.segment = segment;
            this.theta = theta;
            this.z = z;
        }
        public float getTheta() {
            return this.theta;
        }
        public CylinderSegment getSegment() {
            return this.segment;
        }
        public float getZ() {
            return this.z;
        }

        public String toString() {
            return "ModelLocation{Seg: " + segment.toString() + ", theta: " + Float.toString(theta) + ", z: " + Float.toString(z) + "}";
        }

    }

    private static class QuaternionGradient {
        public float x, y, z, w;

        public QuaternionGradient(QuaternionGradient old) {
            this.x = old.x;
            this.y = old.y;
            this.z = old.z;
            this.w = old.w;
        }
        public QuaternionGradient() {
            this.x = this.y = this.z = this.w = 0.0f;
        }
        public QuaternionGradient(float[] init) {
            this(init[0], init[1], init[2], init[3]);
        }
        public QuaternionGradient(float x, float y, float z, float w) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        public QuaternionGradient add(QuaternionGradient other) {
            this.x += other.x;
            this.y += other.y;
            this.z += other.z;
            this.w += other.w;
            return this;
        }

        public QuaternionGradient scale(float s) {
            this.x *= s;
            this.y *= s;
            this.z *= s;
            this.w *= s;
            return this;
        }
        public static QuaternionGradient add(QuaternionGradient one, QuaternionGradient two) {
            return (new QuaternionGradient(one)).add(two);
        }

        /*
        Adjust a quaternion and normalize
         */
        public Quaternion adjustQuaternion(Quaternion q) {
            q.set(q.x + this.x, q.y + this.y, q.z + this.z, q.w + this.w);
            return q;
        }
    }

    /*
    Same deal as iterativeFit, but minimize the distance between model locations and selected _rays_
     */
    public void iterativeRayFit(int num_steps, List<ModelLocation> modelLocations, List<Ray> targets, float[] targetWeights) {
        for (int i = 0; i < num_steps; i++) {
            this.stepTranslateToMatchRays(modelLocations, targets, targetWeights);
            this.stepRotateToMatchRays(modelLocations, targets, targetWeights);
        }
    }

    public void iterativeRayFit(int num_steps, float uniform_step_size_mul, List<RayCorrespondence> correspondences) {
        List<ModelLocation> modelLocations = new ArrayList<>();
        List<Ray> rays = new ArrayList<>();
        float[] weights = new float[correspondences.size()];
        int i = 0;
        for (RayCorrespondence correspondence : correspondences) {
            modelLocations.add(correspondence.modelLocation);
            rays.add(correspondence.targetRay);
            weights[i] = correspondence.targetWeight;
            i += 1;
        }
        iterativeRayFit(num_steps, uniform_step_size_mul, modelLocations, rays, weights);
    }

    public void iterativeRayFit(int num_steps, float uniform_step_size_mul, List<ModelLocation> modelLocations, List<Ray> targets, float[] targetWeights) {
        float[] weights = new float[modelLocations.size()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = uniform_step_size_mul * targetWeights[i];
        }
        iterativeRayFit(num_steps, modelLocations, targets, weights);
    }

    public void iterativeRayFit(int num_steps, float uniform_step_size, List<ModelLocation> modelLocations, List<Ray> targets) {
        float[] weights = new float[modelLocations.size()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = uniform_step_size;
        }
        iterativeRayFit(num_steps, modelLocations, targets, weights);
    }

    public void iterativeFit(int num_steps, float uniform_step_size_mul, List<ModelLocation> modelLocations, List<Vector3> targets, float[] targetWeights) {
        float[] weights = new float[modelLocations.size()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = uniform_step_size_mul * targetWeights[i];
        }
        iterativeFit(num_steps, modelLocations, targets, weights);
    }

    public void iterativeFit(int num_steps, float uniform_step_size, List<ModelLocation> modelLocations, List<Vector3> targets) {
        float[] weights = new float[modelLocations.size()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = uniform_step_size;
        }
        iterativeFit(num_steps, modelLocations, targets, weights);
    }

    public void iterativeFit(int num_steps, List<ModelLocation> modelLocations, List<Vector3> targets, float[] targetWeights) {
        for (int i = 0; i < num_steps; i++) {
            this.jumpTranslateToMatch(modelLocations, targets, targetWeights);
            this.stepRotateToMatch(modelLocations, targets, targetWeights);
        }
    }


    public void stepRotateToMatchRays(List<ModelLocation> modelLocations, List<Ray> targets, float[] targetWeights) {
        for (CylinderSegment segment : CylinderSegment.values()) {
            QuaternionGradient totalGradient = new QuaternionGradient();
            //Perform the adjustment for each segment
            for (int i = 0; i < targets.size(); i++) {
                float weight = targetWeights[i];
                ModelLocation modelLocation = modelLocations.get(i);
                Ray target = targets.get(i);
                QuaternionGradient gradientComponent = getRayAlignGradientForQuat(segment, modelLocation, target);
                gradientComponent.scale(weight);
                totalGradient.add(gradientComponent);
            }
            //Okay, the total gradient has been computed, so now use that to adjust the quaternion
            totalGradient.adjustQuaternion(this.cylinderQuaternions.get(segment.getIndex()));
        }
    }

    public void stepTranslateToMatchRays(List<ModelLocation> modelLocations, List<Ray> targets, float[] targetWeights) {
        Vector3 totalGradient = new Vector3();

        for (int i = 0; i < targets.size(); i++) {
            float weight = targetWeights[i];
            ModelLocation modelLocation = modelLocations.get(i);
            Ray target = targets.get(i);
            Vector3 gradientComponent = getRayAlignGradientForTranslate(modelLocation, target);
            gradientComponent = gradientComponent.scaled(weight);
            totalGradient = Vector3.add(totalGradient, gradientComponent);
        }
        this.initialTranslation = Vector3.add(this.initialTranslation, totalGradient);
    }


    public static class RayCorrespondence {
        ModelLocation modelLocation;
        Ray targetRay;
        float targetWeight;
        public RayCorrespondence(ModelLocation modelLocation, Ray targetRay, float targetWeight) {
            this.modelLocation = modelLocation;
            this.targetRay = targetRay;
            this.targetWeight = targetWeight;
        }
        @Override
        public String toString() {
            return "RayCorrespondence{ModelLocation: " + modelLocation.toString() +
                    ", TargetRay: " + targetRay.toString() + ", TargetWeight: " + Float.toString(targetWeight) + "}";
        }
    }

    /*
    Given parallel lists of model locations, global target points, and weights to lend each target,

     */
    public void stepRotateToMatch(List<ModelLocation> modelLocations, List<Vector3> targets, float[] targetWeights) {
        for (CylinderSegment segment : CylinderSegment.values()) {
            QuaternionGradient totalGradient = new QuaternionGradient();
            //Perform the adjustment for each segment
            for (int i = 0; i < targets.size(); i++) {
                float weight = targetWeights[i];
                ModelLocation modelLocation = modelLocations.get(i);
                Vector3 target = targets.get(i);
                QuaternionGradient gradientComponent = getGlobalAlignGradientForQuat(segment, modelLocation, target);
                gradientComponent.scale(weight);
                totalGradient.add(gradientComponent);
            }
            //Okay, the total gradient has been computed, so now use that to adjust the quaternion
            totalGradient.adjustQuaternion(this.cylinderQuaternions.get(segment.getIndex()));
        }
    }

    /*
    Instantly adjust the global translation of the cylinder model so as to best match the desired target locations
     */
    public void jumpTranslateToMatch(List<ModelLocation> modelLocations, List<Vector3> targets, float[] targetWeights) {
        Vector3 actualWeightedCentroid = new Vector3();
        Vector3 desiredWeightedCentroid = new Vector3();
        float weightSum = 0.0f;
        for (int i = 0; i < targetWeights.length; i++) {
            float weight = targetWeights[i];
            weightSum += weight;
            ModelLocation modelLocation = modelLocations.get(i);

            Vector3 actual = getGlobalPointAt(modelLocation);
            Vector3 target = targets.get(i);

            actualWeightedCentroid = Vector3.add(actualWeightedCentroid, actual.scaled(weight));
            desiredWeightedCentroid = Vector3.add(desiredWeightedCentroid, target.scaled(weight));
        }
        if (weightSum == 0f) {
            return;
        }
        float weightRecip = 1.0f / weightSum;
        Vector3 weightedDiff = Vector3.subtract(desiredWeightedCentroid, actualWeightedCentroid);
        Vector3 diffToApply = weightedDiff.scaled(weightRecip);
        this.initialTranslation = Vector3.add(this.initialTranslation, diffToApply);

    }

    private Vector3 getRayAlignGradientForTranslate(ModelLocation modelLocation, Ray target) {
        float epsilon = 0.01f;

        Vector3 initialTranslation = this.initialTranslation;

        Vector3 initEndPos = getGlobalPointAt(modelLocation);

        Vector3 initPerp = getRayPerp(target, initEndPos);

        Vector3[] elemDirections = epsilonOffset(initialTranslation, epsilon);

        float[] result = {0.0f, 0.0f, 0.0f};
        for (int i = 0; i < elemDirections.length; i++) {
            this.initialTranslation = elemDirections[i];


            Vector3 tempEndPos = getGlobalPointAt(modelLocation);
            Vector3 tempDiff = Vector3.subtract(tempEndPos, initEndPos);
            float improvement = Vector3.dot(initPerp, tempDiff);
            //Log.e(TAG, "Improvement " + Float.toString(improvement));
            result[i] = improvement / epsilon;
        }

        this.initialTranslation = initialTranslation;

        return new Vector3(result[0], result[1], result[2]);
    }

    /*
    Gets a vector perpendicular to the ray origin -> initial end pos
    vector which points toward the ray normal vector emanating from the ray origin
     */
    private Vector3 getRayPerp(Ray target, Vector3 initEndPos) {


        Vector3 initDiff = Vector3.subtract(initEndPos, target.getOrigin());

        Vector3 diffNormal = (new Vector3(initDiff)).normalized();

        Vector3 rayNormal = target.getDirection().normalized();

        Vector3 outOfPlane = Vector3.cross(diffNormal, rayNormal);

        Vector3 result = Vector3.cross(outOfPlane, initDiff);

        return result;
    }

    private QuaternionGradient getRayAlignGradientForQuat(CylinderSegment segmentToRotate, ModelLocation modelLocation, Ray target) {
        float epsilon = 0.01f;

        int segmentInd = segmentToRotate.getIndex();

        Quaternion originalRotation = this.cylinderQuaternions.get(segmentInd);

        Vector3 initEndPos = getGlobalPointAt(modelLocation);

        Vector3 initPerp = getRayPerp(target, initEndPos);

        Quaternion[] elemDirections = epilonOffset(originalRotation, epsilon);

        float[] result = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int i = 0; i < elemDirections.length; i++) {
            this.cylinderQuaternions.set(segmentInd, elemDirections[i]);

            Vector3 tempEndPos = getGlobalPointAt(modelLocation);
            Vector3 tempDiff = Vector3.subtract(tempEndPos, initEndPos);
            float improvement = Vector3.dot(initPerp, tempDiff);
            //Log.e(TAG, "Improvement " + Float.toString(improvement));
            result[i] = improvement / epsilon;
        }

        this.cylinderQuaternions.set(segmentInd, originalRotation);
        return new QuaternionGradient(result);
    }

    private Vector3[] epsilonOffset(Vector3 originalTranslation, float epsilon) {
        Vector3 modOne = new Vector3(originalTranslation);
        modOne.x += epsilon;

        Vector3 modTwo = new Vector3(originalTranslation);
        modTwo.y += epsilon;

        Vector3 modThree = new Vector3(originalTranslation);
        modThree.z += epsilon;

        return new Vector3[]{modOne, modTwo, modThree};
    }

    private Quaternion[] epilonOffset(Quaternion originalRotation, float epsilon) {
        Quaternion modOne = new Quaternion(originalRotation);
        modOne.x += epsilon;
        modOne = modOne.normalized();

        Quaternion modTwo = new Quaternion(originalRotation);
        modTwo.y += epsilon;
        modTwo = modTwo.normalized();

        Quaternion modThree = new Quaternion(originalRotation);
        modThree.z += epsilon;
        modThree = modThree.normalized();


        Quaternion modFour = new Quaternion(originalRotation);
        modFour.w += epsilon;
        modFour = modFour.normalized();

        Quaternion[] elemDirections = {modOne, modTwo, modThree, modFour};
        return elemDirections;
    }

    /*
    For a given cylinder segment to adjust the rotation of, and a model location,
    determine the Quaternionic gradient w.r.t. squared L2 distance
    between the global point corresponding to the model location and the target point
     */
    private QuaternionGradient getGlobalAlignGradientForQuat(CylinderSegment segmentToRotate, ModelLocation modelLocation, Vector3 target) {
        float epsilon = 0.01f;

        int segmentInd = segmentToRotate.getIndex();

        Quaternion originalRotation = this.cylinderQuaternions.get(segmentInd);

        Vector3 initEndPos = getGlobalPointAt(modelLocation);
        Vector3 desiredDiff = Vector3.subtract(target, initEndPos);

        Quaternion[] elemDirections = epilonOffset(originalRotation, epsilon);

        float[] result = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int i = 0; i < elemDirections.length; i++) {
            this.cylinderQuaternions.set(segmentInd, elemDirections[i]);

            Vector3 tempEndPos = getGlobalPointAt(modelLocation);
            Vector3 tempDiff = Vector3.subtract(tempEndPos, initEndPos);
            float improvement = Vector3.dot(desiredDiff, tempDiff);
            //Log.e(TAG, "Improvement " + Float.toString(improvement));
            result[i] = improvement / epsilon;
        }

        this.cylinderQuaternions.set(segmentInd, originalRotation);
        return new QuaternionGradient(result);
    }




}
