import javafx.util.Pair;
import preprocessing.TrainTestSplitting;
import recommendation.CoTraining;
import recommendation.ItemKNN;
import recommendation.UserKNN;
import utility.DataUtility;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        String[] datasets = {"ml-1m"};
        int[] lims = {-1};
        float eval;
        Pair<Float, Float> preRe;
        for (String dataset: datasets) {
            for (int lim: lims) {
                String dataFile = dataset + ".data";
                TrainTestSplitting splitting = new TrainTestSplitting(dataFile, dataset, lim, 26051996);
                splitting.split();
//                String dataDir = splitting.getDataDir();
//                float mi = 0;
//                float mu = 0;
//                float mci = 0;
//                float mcu = 0;
//
//                float ti = 0, pi = 0;
//                float tu = 0, pu = 0;
//                float tci = 0, pci = 0;
//                float tcu = 0, pcu = 0;
//
//                float prei = 0, rei = 0;
//                float preu = 0, reu = 0;
//                float preci = 0, reci = 0;
//                float precu = 0, recu = 0;
//                for (int i = 0; i < 5; i++) {
//                    String foldDir = dataDir + "/" + i;
//                    String trainingFile = foldDir + "/training.txt";
//                    String testingFile = foldDir + "/testing.txt";
//
//                    ItemKNN itemKNN = new ItemKNN(trainingFile, testingFile);
//                    itemKNN.compute(true);
//                    String predictItemFile = foldDir + "/predict_item.txt";
//                    DataUtility.writeData(itemKNN.getPrediction(), predictItemFile, false);
//                    eval = DataUtility.mae(predictItemFile, testingFile);
//                    preRe = DataUtility.precisionRecall(predictItemFile, testingFile);
//                    mi += eval;
//                    ti += itemKNN.getTrainingTime();
//                    pi += itemKNN.getPredictionTime();
//                    prei += preRe.getKey();
//                    rei += preRe.getValue();
//
//                    UserKNN userKNN = new UserKNN(trainingFile, testingFile);
//                    userKNN.compute(true);
//                    String predictUserFile = foldDir + "/predict_user.txt";
//                    DataUtility.writeData(userKNN.getPrediction(), predictUserFile, false);
//                    eval = DataUtility.mae(predictUserFile, testingFile);
//                    preRe = DataUtility.precisionRecall(predictUserFile, testingFile);
//                    mu += eval;
//                    tu += userKNN.getTrainingTime();
//                    pu += userKNN.getPredictionTime();
//                    preu += preRe.getKey();
//                    reu += preRe.getValue();
//
//                    CoTraining coTraining = new CoTraining(trainingFile, testingFile, true);
//                    coTraining.compute();
//                    String predictCotrainingItemFile = foldDir + "/predict_cotraining_item.txt";
//                    DataUtility.writeData(coTraining.getPrediction(), predictCotrainingItemFile, false);
//                    eval = DataUtility.mae(predictCotrainingItemFile, testingFile);
//                    preRe = DataUtility.precisionRecall(predictCotrainingItemFile, testingFile);
//                    mci += eval;
//                    tci += coTraining.getTrainingTime();
//                    pci += coTraining.getPredictionTime();
//                    preci += preRe.getKey();
//                    reci += preRe.getValue();
//
//                    coTraining = new CoTraining(trainingFile, testingFile, false);
//                    coTraining.compute();
//                    String predictCotrainingUserFile = foldDir + "/predict_cotraining_user.txt";
//                    DataUtility.writeData(coTraining.getPrediction(), predictCotrainingUserFile, false);
//                    eval = DataUtility.mae(predictCotrainingUserFile, testingFile);
//                    preRe = DataUtility.precisionRecall(predictCotrainingUserFile, testingFile);
//                    mcu += eval;
//                    tcu += coTraining.getTrainingTime();
//                    pcu += coTraining.getPredictionTime();
//                    precu += preRe.getKey();
//                    recu += preRe.getValue();
//
//                }
//
//                BufferedWriter out = new BufferedWriter(new FileWriter(dataDir + "/evaluation.txt"));
//                out.write("mi: " + mi / 5.0f + "\n");
//                out.write("ti: " + ti / 5.0f + "\n");
//                out.write("pi: " + pi / 5.0f + "\n");
//                out.write("prei: " + prei / 5.0f + "\n");
//                out.write("rei: " + rei / 5.0f + "\n");
//
//                out.write("\n");
//
//                out.write("mu: " + mu / 5.0f + "\n");
//                out.write("tu: " + tu / 5.0f + "\n");
//                out.write("pu: " + pu / 5.0f + "\n");
//                out.write("preu: " + preu / 5.0f + "\n");
//                out.write("reu: " + reu / 5.0f + "\n");
//
//                out.write("\n");
//
//                out.write("mci: " + mci / 5.0f + "\n");
//                out.write("tci: " + tci / 5.0f + "\n");
//                out.write("pci: " + pci / 5.0f + "\n");
//                out.write("preci: " + preci / 5.0f + "\n");
//                out.write("reci: " + reci / 5.0f + "\n");
//
//                out.write("\n");
//
//                out.write("mcu: " + mcu / 5.0f + "\n");
//                out.write("tcu: " + tcu / 5.0f + "\n");
//                out.write("pcu: " + pcu / 5.0f + "\n");
//                out.write("precu: " + precu / 5.0f + "\n");
//                out.write("recu: " + recu / 5.0f + "\n");
//                out.close();
            }
        }

    }
}
