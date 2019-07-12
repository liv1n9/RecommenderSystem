package recommendation;

import javafx.util.Pair;
import utility.Const;
import utility.DataUtility;

import java.util.ArrayList;
import java.util.Map;
import java.util.TreeMap;

public class ItemKNN {
    private String trainingFile;
    private String testingFile;

    private Data data;
    private Map<Pair<Integer, Integer>, Float> itemSimilarity;
    private Map<Integer, Map<Integer, Float>> prediction;
    private Map<Integer, Float> baselineUser;
    private Map<Integer, Float> baselineItem;
    private float trainingTime;
    private float predictionTime;
    private float trainingSimilarityTime;
    private int kNearest;

    public ItemKNN(String trainingFile, String testingFile) {
        this.trainingFile = trainingFile;
        this.testingFile = testingFile;
        data = new Data(trainingFile, testingFile);
        kNearest = (int) Math.sqrt(data.getItems().size());
        itemSimilarity = new TreeMap<>(DataUtility.getPairComparator());
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
        Map<Integer, Map<Integer, Float>> itemRatings = data.getTrainingItemRatings();
        prediction = new TreeMap<>();
        float meanRating = data.getMeanRating();
        for (Map.Entry<Integer, Map<Integer, Float>> entry1 : data.getTestingUserRating().entrySet()) {
            int u = entry1.getKey();
            float baselineU = baselineUser.getOrDefault(u, 0.0f);
            Map<Integer, Float> ratings = entry1.getValue();
            prediction.put(u, new TreeMap<>());
            for (Map.Entry<Integer, Float> entry2 : ratings.entrySet()) {
                int item = entry2.getKey();
                ArrayList<Integer> neighbors = new ArrayList<>();
                for (Map.Entry<Integer, Float> e : data.getTrainingUserRatings().getOrDefault(u, new TreeMap<>()).entrySet()) {
                    int j = e.getKey();
                    neighbors.add(j);
                }
                neighbors.sort((o1, o2) -> Float.compare(similarity(o2, item), similarity(o1, item)));
                float x = 0;
                float y = 0;
                for (int i = 0; i < Math.min(neighbors.size(), kNearest); i++) {
                    int j = neighbors.get(i);
                    float s = similarity(item, j);
                    float r = itemRatings.get(j).get(u) - meanRating - baselineItem.get(j) - baselineU;
                    x += s * r;
                    y += s;
                }
                if (y == 0) {
                    y = 1e-5f;
                }
                float result = meanRating + baselineItem.getOrDefault(item, 0.0f) + baselineU + x / y;
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

    private float similarity(int i, int j) {
        long duration = System.nanoTime();
        if (i > j) {
            int t = i; i = j; j = t;
        }
        Pair<Integer, Integer> key = new Pair<>(i, j);
        if (itemSimilarity.containsKey(key)) {
            duration = System.nanoTime() - duration;
            trainingSimilarityTime += duration / 1e9;
            return itemSimilarity.get(key);
        }
        Map<Integer, Map<Integer, Float>> itemRatings = data.getTrainingItemRatings();
        Map<Integer, Float> firstMap = data.getTrainingItemRatings().getOrDefault(i, new TreeMap<>());
        Map<Integer, Float> secondMap = data.getTrainingItemRatings().getOrDefault(j, new TreeMap<>());
        if (firstMap.size() > secondMap.size()) {
            Map<Integer, Float> temp = firstMap;
            firstMap = secondMap;
            secondMap = temp;
            int t = i; i = j; j = t;
        }
        float x = 0;
        float y1 = 0;
        float y2 = 0;
        int n = 0;
        for (Map.Entry<Integer, Float> entry : firstMap.entrySet()) {
            int user = entry.getKey();
            if (secondMap.containsKey(user)) {
                float riu = entry.getValue();
                float rju = itemRatings.get(j).get(user);
                float deltaI = riu - data.getMeanTrainingItemRating().get(i);
                float deltaJ = rju - data.getMeanTrainingItemRating().get(j);
                x += deltaI * deltaJ;
                y1 += deltaI * deltaI;
                y2 += deltaJ * deltaJ;
                n++;
            }
        }
        float y = (float) Math.sqrt(y1 * y2);
        if (y == 0) {
            y = 1e-5f;
        }
        float pearson = x / y;
        float result = n * (n + 100.0f) * pearson;
        itemSimilarity.put(key, result);
        duration = System.nanoTime() - duration;
        trainingSimilarityTime += duration / 1e9;
        return result;
    }

    public Map<Integer, Map<Integer, Float>> getPrediction() {
        return prediction;
    }

    public Data getData() {
        return data;
    }

    public Map<Integer, Float> getBaselineUser() {
        return baselineUser;
    }

    public Map<Integer, Float> getBaselineItem() {
        return baselineItem;
    }

    public float getTrainingTime() {
        return trainingTime;
    }

    public float getPredictionTime() {
        return predictionTime;
    }
}
