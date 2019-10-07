package com.funktorreactive.animereal;

import com.google.ar.core.Session;
import com.google.ar.sceneform.ux.ArFragment;

import java.util.HashSet;
import java.util.Set;

public class SharedCameraArFragment extends ArFragment {
    public SharedCameraArFragment() {
        super();
    }

    @Override
    public Set<Session.Feature> getSessionFeatures() {
        Set<Session.Feature> defaults = new HashSet<>(super.getSessionFeatures());
        defaults.add(Session.Feature.SHARED_CAMERA);
        return defaults;
    }
}
