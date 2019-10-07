package com.funktorreactive.animereal;

import com.google.ar.sceneform.math.Vector3;

import java.util.ArrayList;
import java.util.List;

public class NetAnnotation {
    public static BodyNet<List<ScreenPointAnnotation>> createBodyNet(StandardBody body, Iterable<ScreenPointAnnotation> annotationList) {
        BodyNet<List<ScreenPointAnnotation>> result = new BodyNet<>(body, () -> new ArrayList<>());
        for (ScreenPointAnnotation annotation : annotationList) {
            Vector3 position = annotation.getEstimatedTemplatePoint();
            result.mapOverPosition(position, (oldList) -> {oldList.add(annotation); return oldList;});
        }
        return result;
    }


}
