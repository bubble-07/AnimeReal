package com.funktorreactive.animereal;

import com.google.ar.sceneform.math.Vector3;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/*
A generic class which associates to every point in a StandardBody
an object of type T, and provides methods to manipulate such structures
locally and globally
 */
public class BodyNet<T> {
    StandardBody template = null;

    private List<T> data = new ArrayList<>();

    private BodyNet(StandardBody template, List<T> data) {
        this.template = template;
        this.data = data;
    }

    public BodyNet(StandardBody template, Supplier<T> dataInitializer) {
        this.template = template;
        init(dataInitializer);
    }
    private void init(Supplier<T> initializer) {
        int num_entries = this.template.getNumPoints();
        for (int i = 0; i < num_entries; i++) {
            data.add(initializer.get());
        }
    }

    public List<T> getTemplateParallelIndexedData() {
        return this.data;
    }

    public T getData(Vector3 closestPoint) {
        int index = template.getNearestIndex(closestPoint);
        return data.get(index);
    }

    public T getDataAtNormalized(Vector3 closestPoint) {
        int index = template.getNearestIndexToNormalized(closestPoint);
        return data.get(index);
    }

    private List<T> getNeighborData(int ind) {
        int[] neighbors = template.getNeighborIndices(ind);
        List<T> result = new ArrayList<>();
        for (int i = 0; i < neighbors.length; i++) {
            int neighbInd = neighbors[i];
            result.add(data.get(neighbInd));
        }
        return result;
    }

    public BodyNet<T> mapOverPosition(Vector3 position, Function<T, T> transformer) {
        int index = template.getNearestIndex(position);
        T data = this.data.get(index);
        T result = transformer.apply(data);
        this.data.set(index, data);
        return this;
    }

    public <R> BodyNet<R> map(Function<T, R> transformer) {
        List<R> mapped = data.stream().map(transformer).collect(Collectors.toList());
        return new BodyNet<>(template, mapped);
    }

    public <R> BodyNet<R> mapNeighborhoods(Function<List<T>, R>  mapper) {
        List<R> resultData = new ArrayList<>();
        int num_entries = this.template.getNumPoints();
        for (int i = 0; i < num_entries; i++) {
            List<T> neighborData = getNeighborData(i);
            resultData.add(mapper.apply(neighborData));
        }
        return new BodyNet<>(template, resultData);
    }

    public BodyNet<T> transform(Function<T, T> transformer) {
        this.data = this.data.stream().map(transformer).collect(Collectors.toList());
        return this;
    }

    public BodyNet<T> transformNeighborhoods(Function<List<T>, T> transformer) {
        int num_entries = this.template.getNumPoints();
        for (int i = 0; i < num_entries; i++) {
            List<T> neighborData = getNeighborData(i);
            T result = transformer.apply(neighborData);
            this.data.set(i, result);
        }
        return this;
    }

    public BodyNet<T> iterateNeighborhoodTransform(Function<List<T>, T> transformer, int num_times) {
        for (int i = 0; i < num_times; i++) {
            this.transformNeighborhoods(transformer);
        }
        return this;
    }

}
