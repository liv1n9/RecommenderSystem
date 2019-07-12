package recommendation;

import javafx.util.Pair;
import utility.DataUtility;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Map;
import java.util.TreeMap;

public class CoTraining {
    private String trainingFile;
    private String testingFile;
    private boolean itemFirst;

    private float trainingTime;
    private float predictionTime;
    private int m;

    private Map<Integer, Map<Integer, Float>> prediction;
    private Map<Integer, Map<Integer, Float>> unlabeledSet;
    private String labeledFile;
    private String[] labeledFiles;
    private String unlabeledFile;

    public CoTraining(String trainingFile, String testingFile, boolean itemFirst) {
        this.trainingFile = trainingFile;
        this.testingFile = testingFile;
        this.itemFirst = itemFirst;
    }

    public void compute() {
        try {
            initData();
            prediction = new TreeMap<>();
            for (int iterator = 0; iterator < 10 && !unlabeledSet.isEmpty(); iterator++) {
                System.out.println("Iteration: " + iterator);
                if (itemFirst) {
                    computeItem();
                    computeUser();
                } else {
                    computeUser();
                    computeItem();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Training time: " + trainingTime);
        System.out.println("Prediction time: " + predictionTime);
    }

    private void computeItem() throws IOException {
        ItemKNN itemKNN = new ItemKNN(labeledFiles[1], unlabeledFile);
        itemKNN.compute(false);

        long duration = System.nanoTime();
        Data itemData = itemKNN.getData();
        Map<Pair<Integer, Integer>, Float> confidentMap = new TreeMap<>(DataUtility.getPairComparator());
        ArrayList<Pair<Integer, Integer>> candidates = new ArrayList<>();
        for (Map.Entry<Integer, Map<Integer, Float>> e : itemKNN.getPrediction().entrySet()) {
            int u = e.getKey();
            int nu = itemData.getTrainingUserRatings().get(u).size();
            float baselineU = itemKNN.getBaselineUser().getOrDefault(u, 0.0f);
            Map<Integer, Float> ratings = e.getValue();
            for (Map.Entry<Integer, Float> entry : ratings.entrySet()) {
                int i = entry.getKey();
                float r = entry.getValue();
                float b = itemData.getMeanRating() + itemKNN.getBaselineItem().getOrDefault(i, 0.0f) + baselineU;
                int ni = itemData.getTrainingItemRatings().getOrDefault(i, new TreeMap<>()).size();
                float conf = nu * ni / Math.abs(b - r);
                Pair<Integer, Integer> key = new Pair<>(u, i);
                confidentMap.put(key, conf);
                candidates.add(key);
            }
        }
        trainingTime += (System.nanoTime() - duration) / 1e9;

        duration = System.nanoTime();
        candidates.sort((o1, o2) -> Float.compare(confidentMap.get(o2), confidentMap.get(o1)));
        Map<Integer, Map<Integer, Float>> predictionSet = new TreeMap<>();
        for (int j = 0; j < Math.min(candidates.size(), this.m); j++) {
            Pair<Integer, Integer> key = candidates.get(j);
            int u = key.getKey();
            int i = key.getValue();
            if (!predictionSet.containsKey(u)) {
                predictionSet.put(u, new TreeMap<>());
            }
            predictionSet.get(u).put(i, itemKNN.getPrediction().get(u).get(i));
            unlabeledSet.get(u).remove(i);
        }
        predictionSet.forEach((u, ratings) -> {
            if (!prediction.containsKey(u)) {
                prediction.put(u, new TreeMap<>());
            }
            ratings.forEach((i, r) -> prediction.get(u).put(i, r));
        });

        predictionTime += (System.nanoTime() - duration) / 1e9;
        DataUtility.writeData(predictionSet, labeledFiles[0], true);
        DataUtility.writeData(unlabeledSet, unlabeledFile, false);
        trainingTime += itemKNN.getTrainingTime();
        predictionTime += itemKNN.getPredictionTime();
    }

    private void computeUser() throws IOException {
        UserKNN userKNN = new UserKNN(labeledFiles[0], unlabeledFile);
        userKNN.compute(false);

        long duration = System.nanoTime();
        Data userData = userKNN.getData();
        Map<Pair<Integer, Integer>, Float> confidentMap = new TreeMap<>(DataUtility.getPairComparator());
        ArrayList<Pair<Integer, Integer>> candidates = new ArrayList<>();
        for (Map.Entry<Integer, Map<Integer, Float>> e : userKNN.getPrediction().entrySet()) {
            int u = e.getKey();
            int nu = userData.getTrainingUserRatings().get(u).size();
            float baselineU = userKNN.getBaselineUser().getOrDefault(u, 0.0f);
            Map<Integer, Float> ratings = e.getValue();
            for (Map.Entry<Integer, Float> entry : ratings.entrySet()) {
                int i = entry.getKey();
                float r = entry.getValue();
                float b = userData.getMeanRating() + userKNN.getBaselineItem().getOrDefault(i, 0.0f) + baselineU;
                int ni = userData.getTrainingItemRatings().getOrDefault(i, new TreeMap<>()).size();
                float conf = nu * ni / Math.abs(b - r);
                Pair<Integer, Integer> key = new Pair<>(u, i);
                confidentMap.put(key, conf);
                candidates.add(key);
            }
        }
        trainingTime += (System.nanoTime() - duration) / 1e9;

        duration = System.nanoTime();
        candidates.sort((o1, o2) -> Float.compare(confidentMap.get(o2), confidentMap.get(o1)));
        Map<Integer, Map<Integer, Float>> predictionSet = new TreeMap<>();
        for (int j = 0; j < Math.min(candidates.size(), this.m); j++) {
            Pair<Integer, Integer> key = candidates.get(j);
            int u = key.getKey();
            int i = key.getValue();
            if (!predictionSet.containsKey(u)) {
                predictionSet.put(u, new TreeMap<>());
            }
            predictionSet.get(u).put(i, userKNN.getPrediction().get(u).get(i));
            unlabeledSet.get(u).remove(i);
        }
        predictionSet.forEach((u, ratings) -> {
            if (!prediction.containsKey(u)) {
                prediction.put(u, new TreeMap<>());
            }
            ratings.forEach((i, r) -> prediction.get(u).put(i, r));
        });

        predictionTime += (System.nanoTime() - duration) / 1e9;
        DataUtility.writeData(predictionSet, labeledFiles[1], true);
        DataUtility.writeData(unlabeledSet, unlabeledFile, false);
        trainingTime += userKNN.getTrainingTime();
        predictionTime += userKNN.getPredictionTime();
    }

    private void initData() throws IOException {
        String dir = new File(trainingFile).getParent();
        labeledFile = dir + "/labeled.txt";
        unlabeledFile = dir + "/unlabeled.txt";
        labeledFiles = new String[2];
        for (int i = 0; i < 2; i++) {
            labeledFiles[i] = dir + "/labeled_" + i + ".txt";
            Files.copy(new File(trainingFile).toPath(), new File(labeledFiles[i]).toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        Files.copy(new File(trainingFile).toPath(), new File(labeledFile).toPath(), StandardCopyOption.REPLACE_EXISTING);
        Files.copy(new File(testingFile).toPath(), new File(unlabeledFile).toPath(), StandardCopyOption.REPLACE_EXISTING);
        Data data = new Data(trainingFile, testingFile);
        unlabeledSet = data.getTestingUserRating();
        unlabeledSet.forEach((u, ratings) -> m += ratings.size());
        m = (m + 19) / 20;
        prediction = new TreeMap<>();
    }

    public Map<Integer, Map<Integer, Float>> getPrediction() {
        return prediction;
    }

    public float getTrainingTime() {
        return trainingTime;
    }

    public float getPredictionTime() {
        return predictionTime;
    }
}
