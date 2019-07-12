package utility;

import javafx.util.Pair;
import model.Feedback;

import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;

public class DataUtility {
    public static ArrayList<Feedback> readData(String file) throws IOException {
        BufferedReader in = new BufferedReader(new FileReader(file));
        String line;
        ArrayList<Feedback> feedbackList = new ArrayList<>();
        while ((line = in.readLine()) != null) {
            String[] info = line.split("\\s+");
            if (info.length >= 3) {
                int user, item;
                float rating;
                user = Integer.parseInt(info[0]);
                item = Integer.parseInt(info[1]);
                rating = Float.parseFloat(info[2]);
                feedbackList.add(new Feedback(user, item, rating));
            }
        }
        in.close();
        return feedbackList;
    }

    public static Map<Integer, Map<Integer, Float>> getUserRatings(ArrayList<Feedback> feedbackList) {
        Map<Integer, Map<Integer, Float>> userRatings = new TreeMap<>();
        for (Feedback feedback: feedbackList) {
            int user = feedback.getUser();
            int item = feedback.getItem();
            float rating = feedback.getRating();
            if (!userRatings.containsKey(user)) {
                userRatings.put(user, new TreeMap<>());
            }
            userRatings.get(user).put(item, rating);
        }
        return userRatings;
    }

    public static Map<Integer, Map<Integer, Float>> getItemRatings(ArrayList<Feedback> feedbackList) {
        Map<Integer, Map<Integer, Float>> itemRatings = new TreeMap<>();
        for (Feedback feedback: feedbackList) {
            int user = feedback.getUser();
            int item = feedback.getItem();
            float rating = feedback.getRating();
            if (!itemRatings.containsKey(item)) {
                itemRatings.put(item, new TreeMap<>());
            }
            itemRatings.get(item).put(user, rating);
        }
        return itemRatings;
    }

    public static void writeData(Map<Integer, Map<Integer, Float>> data, String file, boolean append) throws IOException {
        BufferedWriter out = new BufferedWriter(new FileWriter(file, append));
        for (Map.Entry<Integer, Map<Integer, Float>> e : data.entrySet()) {
            int u = e.getKey();
            Map<Integer, Float> ratings = e.getValue();
            for (Map.Entry<Integer, Float> entry : ratings.entrySet()) {
                int i = entry.getKey();
                float r = entry.getValue();
                out.flush();
                out.write(u + "\t" + i + "\t" + r + "\n");
            }
        }
        out.close();
    }

    public static Comparator<Pair<Integer, Integer>> getPairComparator() {
        return (o1, o2) -> {
            int x1 = o1.getKey();
            int x2 = o2.getKey();
            int y1 = o1.getValue();
            int y2 = o2.getValue();
            return (x1 != x2) ? Integer.compare(x1, x2) : Integer.compare(y1, y2);
        };
    }

    public static float mae(String predictionFile, String testingFile) throws IOException {
        Map<Integer, Map<Integer, Float>> predictRatings = getUserRatings(readData(predictionFile));
        Map<Integer, Map<Integer, Float>> testRatings = getUserRatings(readData(testingFile));
        float mae = 0;
        for (Map.Entry<Integer, Map<Integer, Float>> entry : predictRatings.entrySet()) {
            int u = entry.getKey();
            float maeu = 0;
            int count = 0;
            Map<Integer, Float> ratings = entry.getValue();
            for (Map.Entry<Integer, Float> e : ratings.entrySet()) {
                int i = e.getKey();
                float r1 = e.getValue();
                float r2 = testRatings.getOrDefault(u, new TreeMap<>()).getOrDefault(i, 0.0f);
                maeu += Math.abs(r1 - r2);
                count++;
            }
            maeu /= count;
            mae += maeu;
        }
        mae /= predictRatings.size();
        return mae;
    }

    public static Pair<Float, Float> precisionRecall(String predictionFile, String testingFile) throws IOException {
        Map<Integer, Map<Integer, Float>> predictRatings = getUserRatings(readData(predictionFile));
        Map<Integer, Map<Integer, Float>> testRatings = getUserRatings(readData(testingFile));
        int a = 0, b = 0, c = 0, d = 0;
        for (Map.Entry<Integer, Map<Integer, Float>> entry : predictRatings.entrySet()) {
            int u = entry.getKey();
            Map<Integer, Float> ratings = entry.getValue();
            for (Map.Entry<Integer, Float> e : ratings.entrySet()) {
                int i = e.getKey();
                float r1 = e.getValue();
                float r2 = testRatings.getOrDefault(u, new TreeMap<>()).getOrDefault(i, 0.0f);
                if (r1 < 3 && r2 < 3) a++;
                if (r1 >= 3 && r2 < 3) b++;
                if (r1 < 3 && r2 >= 3) c++;
                if (r1 >= 3 && r2 >= 3) d++;
            }
        }
        float pre = d * 1.0f / (b + d);
        float re = d * 1.0f / (c + d);
        return new Pair<>(pre, re);
    }
}
