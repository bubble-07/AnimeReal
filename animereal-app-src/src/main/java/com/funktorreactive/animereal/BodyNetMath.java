package com.funktorreactive.animereal;

import com.google.ar.sceneform.collision.Ray;
import com.google.ar.sceneform.math.Vector3;

import java.util.ArrayList;
import java.util.List;

public class BodyNetMath {
    public static List<AnnotatorOutput.SingleObservation> reduceAverage(List<AnnotatorOutput.SingleObservation> list) {
        if (list.size() == 0) {
            return new ArrayList<>();
        }

        Vector3 rayHeading = new Vector3();
        Vector3 templatePos = new Vector3();
        float totalWeight = 0.0f;

        for (AnnotatorOutput.SingleObservation observation : list) {
            float weight = observation.getSegmentationProb();
            totalWeight +=  weight;

            Vector3 headingContrib = observation.getRay().getDirection();
            headingContrib = headingContrib.scaled(weight);

            Vector3 templateContrib = observation.getNormalizedTemplatePos();
            templateContrib = templateContrib.scaled(weight);

            rayHeading = Vector3.add(rayHeading, headingContrib);
            templatePos = Vector3.add(templateContrib, templatePos);
        }
        float invWeight = 1.0f / totalWeight;

        rayHeading = rayHeading.scaled(invWeight);
        templatePos = templatePos.scaled(invWeight);

        AnnotatorOutput.SingleObservation result = new AnnotatorOutput.SingleObservation(totalWeight, templatePos, new Ray(list.get(0).getRay().getOrigin(), rayHeading));
        List<AnnotatorOutput.SingleObservation> resultList = new ArrayList<>();
        resultList.add(result);
        return resultList;
    }
}
