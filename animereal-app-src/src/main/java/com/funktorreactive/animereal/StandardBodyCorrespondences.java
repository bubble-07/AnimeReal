package com.funktorreactive.animereal;

import android.text.style.TabStopSpan;
import android.util.Log;
import android.util.Pair;

import com.google.ar.sceneform.collision.Ray;
import com.google.ar.sceneform.math.Vector3;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import de.biomedical_imaging.edu.wlu.cs.levy.CG.KDTree;

/*
 * Contains correspondences between standard body points and cylinder points
 */
public class StandardBodyCorrespondences {

    private List<BasicCylinderModel.ModelLocation> modelCorrespondences;

    private StandardBody templateBody;

    private StandardBodyCorrespondences(StandardBody templateBody, List<BasicCylinderModel.ModelLocation> modelCorrespondences) {
        this.modelCorrespondences = modelCorrespondences;
        this.templateBody = templateBody;
    }

    public List<BasicCylinderModel.RayCorrespondence> getRayCorrespondences(BodyNet<List<AnnotatorOutput.SingleObservation>> annotatorOutput) {
        List<BasicCylinderModel.RayCorrespondence> result = new ArrayList<>();

        List<List<AnnotatorOutput.SingleObservation>> indexedObservations = annotatorOutput.getTemplateParallelIndexedData();
        for (int i = 0; i < indexedObservations.size(); i++) {
            List<AnnotatorOutput.SingleObservation> nodeObservations = indexedObservations.get(i);

            BasicCylinderModel.ModelLocation modelLocation = modelCorrespondences.get(i);

            //Log.e("AnimeReal", modelLocation.toString());

            for (AnnotatorOutput.SingleObservation nodeObservation : nodeObservations) {
                Ray ray = nodeObservation.getRay();
                float weight = nodeObservation.getSegmentationProb();
                BasicCylinderModel.RayCorrespondence correspondenceResult = new BasicCylinderModel.RayCorrespondence(modelLocation, ray, weight);
                result.add(correspondenceResult);
            }
        }
        return result;

    }


    private static class Correspondence {
        private BasicCylinderModel.ModelLocation modelLocation;
        private Vector3 comparisonImageSpace;
        public Correspondence(BasicCylinderModel.ModelLocation modelLocation, Vector3 comparisonImageSpace) {
            this.modelLocation = modelLocation;
            this.comparisonImageSpace = comparisonImageSpace;
        }
        public BasicCylinderModel.ModelLocation getModelLocation() {
            return this.modelLocation;
        }
        public Vector3 getTemplatePosition(StandardBody body) {
            Vector3 normalized = getNormalizedTemplatePosition();
            return body.normalizedToTemplateCoords(normalized);
        }
        public Vector3 getTemplateImagePosition() {
            return this.comparisonImageSpace;
        }

        public Vector3 getNormalizedTemplatePosition() {
            float x = this.comparisonImageSpace.x;
            float y = this.comparisonImageSpace.y;
            float z = this.comparisonImageSpace.z;

            x = x / max_x;
            y = y / max_y;
            z = z / max_z;

            return new Vector3(x, y, z);
        }
    }
    private static float mid_x = 145.0f;
    private static float neck_base_y = 67.0f;
    private static float max_z = 82.0f;
    private static float back_rot = 3.1415f / 2.0f;
    private static float leg_top_y = 227.0f;
    private static float max_y = 430.0f;
    private static float knee_y = 300.0f;
    private static float knee_front_z = 70.0f;
    private static float knee_back_z = 40.0f;
    private static float elbow_front_z = 32.0f;
    private static float elbow_back_z = 8.0f;
    private static float elbow_y = 100.0f;
    private static float fingertip_y = 18.0f;
    private static float max_x = 293.0f;

    private static float foreleg_front_z = 29.0f;

    public static BasicCylinderModel getFittedCylinderModelForBody(StandardBody body) {
        List<Vector3> targets = new ArrayList<>();
        List<BasicCylinderModel.ModelLocation> locations = new ArrayList<>();

        for (Correspondence c : correspondences) {
            Vector3 targetPos = c.getTemplatePosition(body);
            BasicCylinderModel.ModelLocation modelLocation = c.getModelLocation();

            targets.add(targetPos);
            locations.add(modelLocation);
        }

        BasicCylinderModel result = new BasicCylinderModel();
        //Scale from template in meters to template in mm
        result.scaleDimensions(1000.0f);

        float alpha = 0.01f;
        int num_steps = 100;

        result.iterativeFit(num_steps, alpha, locations, targets);

        return result;
    }

    public static StandardBodyCorrespondences getStandardCorrespondences(StandardBody standardBody) {
        BasicCylinderModel fittedCylinderModel = getFittedCylinderModelForBody(standardBody);
        //Okay, great. Now pick correspondences by associating approximate closest points on the
        //fitted cylinder model with standard body points

        int z_subdivisions = 10;
        int theta_subdivisions = 10;

        KDTree<BasicCylinderModel.ModelLocation> cylinderKdTree = new KDTree<BasicCylinderModel.ModelLocation>(3);

        for (BasicCylinderModel.CylinderSegment segment : BasicCylinderModel.CylinderSegment.values()) {
            for (int z_ind = 0; z_ind < z_subdivisions; z_ind++) {
                for (int theta_ind = 0; theta_ind < theta_subdivisions; theta_ind++) {
                    float z = (1.0f / z_subdivisions) * z_ind;
                    float theta = (6.2832f / theta_subdivisions) * theta_ind;
                    BasicCylinderModel.ModelLocation modelLocation = new BasicCylinderModel.ModelLocation(segment, theta, z);

                    Vector3 location = fittedCylinderModel.getGlobalPointAt(modelLocation);
                    double[] key = new double[]{(double)location.x, (double)location.y, (double)location.z};

                    try {
                        cylinderKdTree.insert(key, modelLocation);
                    }
                    catch (Throwable t) {
                        Log.e("AnimeReal", t.getMessage());
                    }
                }
            }
        }

        List<BasicCylinderModel.ModelLocation> locList = new ArrayList<>();

        for (int i = 0; i < standardBody.getNumPoints(); i++) {
            Vector3 location = standardBody.getPointAtIndex(i);
            double[] key = new double[]{(double)location.x, (double)location.y, (double)location.z};

            try {
                BasicCylinderModel.ModelLocation closeModelLocation = cylinderKdTree.nearest(key);
                locList.add(closeModelLocation);
            }
            catch (Throwable t) {
                Log.e("AnimeReal", t.getMessage());
            }
        }
        return new StandardBodyCorrespondences(standardBody, locList);
    }



    private static void addCorrespondence(BasicCylinderModel.CylinderSegment segment, float theta, float z, float x, float y, float z_loc) {
        Vector3 vector = new Vector3(x, y, z_loc);
        BasicCylinderModel.ModelLocation location = new BasicCylinderModel.ModelLocation(segment, theta, z);
        correspondences.add(new Correspondence(location, vector));
    }

    private static List<Correspondence> correspondences = new ArrayList<>();
    static {
        addCorrespondence(BasicCylinderModel.CylinderSegment.BODY, 0.0f, 1.0f, mid_x, neck_base_y, max_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.BODY, back_rot, 1.0f, mid_x, neck_base_y, 13.0f);

        addCorrespondence(BasicCylinderModel.CylinderSegment.HEAD, 0.0f, 0.0f, mid_x, neck_base_y, max_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.HEAD, back_rot, 0.0f, mid_x, neck_base_y, 20.0f);

        addCorrespondence(BasicCylinderModel.CylinderSegment.HEAD, 0.0f, 1.0f, mid_x, 0.0f, max_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.HEAD, back_rot, 1.0f, mid_x, 0.0f, 20.0f);

        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_LEG, 0.0f, 0.0f, 107.0f, leg_top_y, 61.0f);
        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_LEG, back_rot, 0.0f, 107.0f, leg_top_y, 61.0f);

        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_LEG, 0.0f, 0.0f, 190.0f, leg_top_y, 61.0f);
        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_LEG, back_rot, 0.0f, 190.0f, leg_top_y, 61.0f);

        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_FORELEG, 0.0f, 1.0f, 55.0f, max_y, foreleg_front_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_FORELEG, back_rot, 1.0f, 55.0f, max_y, 0.0f);

        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_FORELEG, 0.0f, 1.0f, 231.0f, max_y, foreleg_front_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_FORELEG, back_rot, 1.0f, 231.0f, max_y, 0.0f);

        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_FORELEG, 0.0f, 0.0f, 75.0f, knee_y, knee_front_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_FORELEG, back_rot, 0.0f, 75.0f, knee_y, knee_back_z);

        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_FORELEG, 0.0f, 0.0f, 75.0f, knee_y, knee_front_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_FORELEG, back_rot, 0.0f, 75.0f, knee_y, knee_back_z);

        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_FORELEG, 0.0f, 0.0f, 215.0f, knee_y, knee_front_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_FORELEG, back_rot, 0.0f, 215.0f, knee_y, knee_back_z);


        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_ARM, 0.0f, 1.0f, 40.0f, elbow_y, elbow_front_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_ARM, back_rot, 1.0f, 40.0f, elbow_y, elbow_back_z);

        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_ARM, 0.0f, 1.0f, 250.0f, elbow_y, elbow_front_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_ARM, back_rot, 1.0f, 250.0f, elbow_y, elbow_back_z);


        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_FOREARM, 0.0f, 0.0f, 40.0f, elbow_y, elbow_front_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_FOREARM, back_rot, 0.0f, 40.0f, elbow_y, elbow_back_z);

        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_FOREARM, 0.0f, 0.0f, 250.0f, elbow_y, elbow_front_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_FOREARM, back_rot, 0.0f, 250.0f, elbow_y, elbow_back_z);

        addCorrespondence(BasicCylinderModel.CylinderSegment.RIGHT_FOREARM, 0.0f, 1.0f, 14.0f, fingertip_y, max_z);
        addCorrespondence(BasicCylinderModel.CylinderSegment.LEFT_FOREARM, 0.0f, 1.0f, 275.0f, fingertip_y, max_z);



    }

}
