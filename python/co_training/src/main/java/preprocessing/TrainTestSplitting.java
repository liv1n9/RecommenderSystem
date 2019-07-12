package preprocessing;

import javafx.util.Pair;
import model.Feedback;
import utility.DataUtility;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class TrainTestSplitting {
    private String file;
    private String datasetName;
    private long seed;
    private int lim;
    private String dataDir;

    public TrainTestSplitting(String file, String datasetName, int lim, long seed) {
        this.file = file;
        this.datasetName = datasetName;
        this.lim = lim;
        this.seed = seed;
        dataDir = datasetName + "/" + lim;
    }

    public void split() throws IOException {
        ArrayList<Feedback> feedbackList = DataUtility.readData(file);
        Map<Integer, Map<Integer, Float>> userRatings = DataUtility.getUserRatings(feedbackList);

        ArrayList<Pair<Integer, Map<Integer, Float>>> data = new ArrayList<>();
        for (Map.Entry<Integer, Map<Integer, Float>> entry: userRatings.entrySet()) {
            data.add(new Pair<>(entry.getKey(), entry.getValue()));
        }
        Collections.shuffle(data, new Random(seed));
        int step = (data.size() + 4) / 5;
        int l = 0;
        for (int fold = 0; fold < 5; fold++) {
            String foldDir = dataDir + "/" + fold;
            File fileDir = new File(foldDir);
            if (!fileDir.exists()) {
                fileDir.mkdirs();
            }
            String trainingFile = foldDir + "/training.txt";
            String testingFile = foldDir + "/testing.txt";
            int r = Math.min(l + step, data.size());
            Map<Integer, Map<Integer, Float>> trainingSet = new TreeMap<>();
            Map<Integer, Map<Integer, Float>> testingSet = new TreeMap<>();
            for (int i = 0; i < data.size(); i++) {
                int u = data.get(i).getKey();
                if (i < l || i >= r) {
                    trainingSet.put(u, data.get(i).getValue());
                    testingSet.put(u, new TreeMap<>());
                } else {
                    trainingSet.put(u, new TreeMap<>());
                    testingSet.put(u, new TreeMap<>());
                    ArrayList<Pair<Integer, Float>> ratings = new ArrayList<>();
                    for (Map.Entry<Integer, Float> entry : data.get(i).getValue().entrySet()) {
                        int item = entry.getKey();
                        float rating = entry.getValue();
                        ratings.add(new Pair<>(item, rating));
                    }
                    Collections.shuffle(ratings, new Random(seed));
                    int limU;
                    if (lim == -1) {
                        limU = (ratings.size() + 1) / 2;
                    } else {
                        limU = lim;
                    }
                    for (int j = 0; j < ratings.size(); j++) {
                        if (j < limU) {
                            trainingSet.get(u).put(ratings.get(j).getKey(), ratings.get(j).getValue());
                        } else {
                            testingSet.get(u).put(ratings.get(j).getKey(), ratings.get(j).getValue());
                        }
                    }
                }
            }
            l = r;
            DataUtility.writeData(trainingSet, trainingFile, false);
            DataUtility.writeData(testingSet, testingFile, false);
        }
    }

    public String getDataDir() {
        return dataDir;
    }
}
