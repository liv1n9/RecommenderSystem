package recommendation;

import model.Feedback;
import utility.DataUtility;

import java.io.IOException;
import java.util.*;

public class Data {
    private String trainingFile;
    private String testingFile;

    private Map<Integer, Map<Integer, Float>> trainingUserRatings;
    private Map<Integer, Map<Integer, Float>> trainingItemRatings;
    private Map<Integer, Map<Integer, Float>> testingUserRating;
    private Map<Integer, Map<Integer, Float>> testingItemRating;
    private Map<Integer, Float> meanTrainingUserRating;
    private Map<Integer, Float> meanTrainingItemRating;
    private Set<Integer> users;
    private Set<Integer> items;
    private float meanRating;

    public Data(String trainingFile, String testingFile) {
        this.trainingFile = trainingFile;
        this.testingFile = testingFile;
        initData();
    }

    private void initData() {
        try {
            ArrayList<Map<Integer, Map<Integer, Float>>> trainingData = readData(trainingFile);
            trainingUserRatings = trainingData.get(0);
            trainingItemRatings = trainingData.get(1);
            ArrayList<Map<Integer, Map<Integer, Float>>> testingData = readData(testingFile);
            testingUserRating = testingData.get(0);
            testingItemRating = testingData.get(1);

            meanTrainingUserRating = new TreeMap<>();
            meanTrainingItemRating = new TreeMap<>();
            users = new TreeSet<>();
            items = new TreeSet<>();

            int total = 0;
            for (Map.Entry<Integer, Map<Integer, Float>> e : trainingUserRatings.entrySet()) {
                int u = e.getKey();
                users.add(u);
                Map<Integer, Float> ratings = e.getValue();
                float mean = 0;
                int count = 0;
                for (Map.Entry<Integer, Float> entry : ratings.entrySet()) {
                    float rating = entry.getValue();
                    meanRating += rating;
                    mean += rating;
                    total++;
                    count++;
                }
                meanTrainingUserRating.put(u, count > 0 ? mean / count : 0);
            }
            meanRating /= total;
            for (Map.Entry<Integer, Map<Integer, Float>> e : trainingItemRatings.entrySet()) {
                int i = e.getKey();
                items.add(i);
                Map<Integer, Float> ratings = e.getValue();
                float mean = 0;
                int count = 0;
                for (Map.Entry<Integer, Float> entry : ratings.entrySet()) {
                    float rating = entry.getValue();
                    mean += rating;
                    count++;
                }
                meanTrainingItemRating.put(i, count > 0 ? mean / count : 0);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private ArrayList<Map<Integer, Map<Integer, Float>>> readData(String file)
            throws IOException {
        ArrayList<Feedback> feedbackList = DataUtility.readData(file);
        Map<Integer, Map<Integer, Float>> userRatings = DataUtility.getUserRatings(feedbackList);
        Map<Integer, Map<Integer, Float>> itemRatings = DataUtility.getItemRatings(feedbackList);
        ArrayList<Map<Integer, Map<Integer, Float>> > result = new ArrayList<>();

        result.add(userRatings);
        result.add(itemRatings);
        return result;
    }

    public Map<Integer, Map<Integer, Float>> getTrainingUserRatings() {
        return trainingUserRatings;
    }

    public Map<Integer, Map<Integer, Float>> getTrainingItemRatings() {
        return trainingItemRatings;
    }

    public Map<Integer, Map<Integer, Float>> getTestingUserRating() {
        return testingUserRating;
    }

    public Map<Integer, Map<Integer, Float>> getTestingItemRating() {
        return testingItemRating;
    }

    public Map<Integer, Float> getMeanTrainingUserRating() {
        return meanTrainingUserRating;
    }

    public Map<Integer, Float> getMeanTrainingItemRating() {
        return meanTrainingItemRating;
    }

    public Set<Integer> getUsers() {
        return users;
    }

    public Set<Integer> getItems() {
        return items;
    }

    public float getMeanRating() {
        return meanRating;
    }
}
