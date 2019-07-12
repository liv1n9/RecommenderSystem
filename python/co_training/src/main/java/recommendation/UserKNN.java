package recommendation;

import javafx.util.Pair;
import utility.Const;
import utility.DataUtility;

import java.util.ArrayList;
import java.util.Map;
import java.util.TreeMap;

public class UserKNN {
    private String trainingFile;
    private String testingFile;

    private Data data;
    private Map<Pair<Integer, Integer>, Float> userSimilarity;
    private Map<Integer, Map<Integer, Float>>prediction;
    private float trainingTime;
    private float predictionTime;
    private float trainingSimilarityTime;
    private Map<Integer, Float> baselineUser;
    private Map<Integer, Float> baselineItem;
    private int kNearest;

    public UserKNN(String trainingFile, String testingFile) {
        this.trainingFile = trainingFile;
        this.testingFile = testingFile;
        data = new Data(trainingFile, testingFile);
        kNearest = (int) Math.sqrt(data.getUsers().size());
        userSimilarity = new TreeMap<>(DataUtility.getPairComparator());
    }

    public void compute(boolean inform) {
        train();
        predict();
        if (inform) {
            System.out.println("Training time: " + trainingTime);
            System.out.println("Prediction time: " + predictionTime);
        }
    }

    private void train() {
        long duration = System.nanoTime();
        trainBaseline();
        duration = System.nanoTime() - duration;
        trainingTime += duration / 1e9;
    }

    private void trainBaseline() {
        baselineItem = new TreeMap<>();
        baselineUser = new TreeMap<>();
        float meanRating = data.getMeanRating();
        for (int iterator = 0; iterator < Const.BASELINE_ITERATION; iterator++) {
            for (int i: data.getItems()) {
                float x = 0;
                int y = Const.LAMBDA_2;
                for (Map.Entry<Integer, Float> entry : data.getTrainingItemRatings().get(i).entrySet()) {
                    int u = entry.getKey();
                    float r = entry.getValue();
                    x += r - meanRating - baselineUser.getOrDefault(u, 0.0f);
                    y++;
                }
                baselineItem.put(i, x / y);
            }
            for (int u: data.getUsers()) {
                float x = 0;
                float y = Const.LAMBDA_3;
                for (Map.Entry<Integer, Float> entry : data.getTrainingUserRatings().get(u).entrySet()) {
                    int i = entry.getKey();
                    float r = entry.getValue();
                    x += r - meanRating - baselineItem.getOrDefault(i, 0.0f);
                    y++;
                }
                baselineUser.put(u, x / y);
            }
        }
    }

    private void predict() {
        long duration = System.nanoTime();
        Map<Integer, Map<Integer, Float>> userRatings = data.getTrainingUserRatings();
        prediction = new TreeMap<>();
        float meanRating = data.getMeanRating();
        for (Map.Entry<Integer, Map<Integer, Float>> entry1 : data.getTestingUserRating().entrySet()) {
            int u = entry1.getKey();
            float baselineU = baselineUser.getOrDefault(u, 0.0f);
            Map<Integer, Float> ratings = entry1.getValue();
            prediction.put(u, new TreeMap<>());
            for (Map.Entry<Integer, Float> entry2 : ratings.entrySet()) {
                int item = entry2.getKey();
                float baselineI = baselineItem.getOrDefault(item, 0.0f);
                ArrayList<Integer> neighbors = new ArrayList<>();
                for (Map.Entry<Integer, Float> e : data.getTrainingItemRatings().getOrDefault(item, new TreeMap<>()).entrySet()) {
                    int v = e.getKey();
                    neighbors.add(v);
                }
                final int U = u;
                neighbors.sort((o1, o2) -> Float.compare(similarity(o2, U), similarity(o1, U)));
                float x = 0;
                float y = 0;
                for (int i = 0; i < Math.min(neighbors.size(), kNearest); i++) {
                    int v = neighbors.get(i);
                    float s = similarity(u, v);
                    float r = userRatings.get(v).get(item) - meanRating - baselineI - baselineUser.get(v);
                    x += s * r;
                    y += s;
                }
                if (y == 0) {
                    y = 1e-5f;
                }
                float result = meanRating + baselineI + baselineU + x / y;
                if (result < 1) {
                    result = 1;
                }
                if (result > 5) {
                    result = 5;
                }
                prediction.get(u).put(item, result);
            }
        }
        duration = System.nanoTime() - duration;
        trainingTime += trainingSimilarityTime;
        predictionTime += duration / 1e9 - trainingSimilarityTime;
    }


    private float similarity(int u, int v) {
        long duration = System.nanoTime();
        if (u > v) {int t = u; u = v; v = t;}
        Pair<Integer, Integer> key = new Pair<>(u, v);
        if (userSimilarity.containsKey(key)) {
            duration = System.nanoTime() - duration;
            trainingSimilarityTime += duration / 1e9;
            return userSimilarity.get(key);
        }
        Map<Integer, Map<Integer, Float>> userRatings = data.getTrainingUserRatings();
        Map<Integer, Float> firstMap = data.getTrainingUserRatings().getOrDefault(u, new TreeMap<>());
        Map<Integer, Float> secondMap = data.getTrainingUserRatings().getOrDefault(v, new TreeMap<>());
        if (firstMap.size() > secondMap.size()) {
            Map<Integer, Float> temp = firstMap;
            firstMap = secondMap;
            secondMap = temp;
            int t = u; u = v; v = t;
        }
        float x = 0;
        float y1 = 0;
        float y2 = 0;
        int n = 0;
        for (Map.Entry<Integer, Float> entry : firstMap.entrySet()) {
            int item = entry.getKey();
            if (secondMap.containsKey(item)) {
                float rui = entry.getValue();
                float rvi = userRatings.get(v).get(item);
                float deltaU = rui - data.getMeanTrainingUserRating().get(u);
                float deltaV = rvi - data.getMeanTrainingUserRating().get(v);
                x += deltaU * deltaV;
                y1 += deltaU * deltaU;
                y2 += deltaV * deltaV;
                n++;
            }
        }
        float y = (float) Math.sqrt(y1 * y2);
        if (y == 0) {
            y = 1e-5f;
        }
        float pearson = x / y;
        float result = n / (n + 100.0f) * pearson;
        userSimilarity.put(key, result);
        duration = System.nanoTime() - duration;
        trainingSimilarityTime += duration / 1e9;
        return result;
    }

    public Data getData() {
        return data;
    }

    public Map<Integer, Float> getBaselineItem() {
        return baselineItem;
    }

    public Map<Integer, Float> getBaselineUser() {
        return baselineUser;
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
