package com.daeho;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.types.StructType;

public class MLR {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("ML").config("spark.master", "local").getOrCreate();

        StructType schema = new StructType()
                .add("index", "int")
                .add("cycle", "double").add("ECG_a_P", "double")
                .add("ECG_P_b", "double").add("ECG_b_c", "double")
                .add("ECG_c_Q", "double").add("ECG_Q_R", "double")
                .add("ECG_R_S", "double").add("ECG_S_d", "double")
                .add("ECG_d_e", "double").add("ECG_e_T", "double")
                .add("ECG_T_f", "double").add("ECG_S_T", "double")
                .add("ECG_P_peak", "double").add("ECG_Q_peak", "double")
                .add("ECG_R_peak", "double").add("ECG_S_peak", "double")
                .add("ECG_T_peak", "double").add("ECG_RRI", "double")
                .add("TYPE", "double").add("PR_interval", "double")
                .add("PR_segment", "double").add("QRS_complex", "double")
                .add("ST_segment", "double").add("QT_interval", "double")
                .add("target", "double").add("label", "double");

        Dataset<Row> indataset = spark.read().option("header", "true").schema(schema).csv("dataset_index.csv");

        ArrayList<String> inputColsList = new ArrayList<String>(Arrays.asList(indataset.columns()));
        System.out.println(inputColsList);

        inputColsList.remove("index");
        inputColsList.remove("target");
        //inputColsList.remove("label");
        System.out.println(inputColsList);

        String[] inputCols = inputColsList.parallelStream().toArray(String[]::new);

        VectorAssembler assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features");
        Dataset<Row> dataset = assembler.transform(indataset);
        dataset.show();

        Dataset<Row>[] set = dataset.randomSplit(new double[] {0.8, 0.2}, 421);
        Dataset<Row> train_set = set[0];
        Dataset<Row> test_set = set[1];

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.03)
                .setElasticNetParam(0.1)
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setWeightCol("target");

        // Fit the model
        LogisticRegressionModel lrModel = lr.fit(train_set);

        Dataset<Row> predicted = lrModel.transform(test_set);
        predicted.select("target", "label", "prediction").show();

        // Print the coefficients and intercept for multinomial logistic regression
        System.out.println("Coefficients: \n"
                + lrModel.coefficientMatrix() + " \nIntercept: " + lrModel.interceptVector());
        LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();

        // Obtain the loss per iteration.
        double[] objectiveHistory = trainingSummary.objectiveHistory();
        for (double lossPerIteration : objectiveHistory) {
            System.out.println(lossPerIteration);
        }

        // for multiclass, we can inspect metrics on a per-label basis
        System.out.println("False positive rate by label:");
        int i = 0;
        double[] fprLabel = trainingSummary.falsePositiveRateByLabel();
        for (double fpr : fprLabel) {
            System.out.println("label " + i + ": " + fpr);
            i++;
        }

        System.out.println("True positive rate by label:");
        i = 0;
        double[] tprLabel = trainingSummary.truePositiveRateByLabel();
        for (double tpr : tprLabel) {
            System.out.println("label " + i + ": " + tpr);
            i++;
        }

        System.out.println("Precision by label:");
        i = 0;
        double[] precLabel = trainingSummary.precisionByLabel();
        for (double prec : precLabel) {
            System.out.println("label " + i + ": " + prec);
            i++;
        } // Precision(정밀도): 모델이 'True'라고 분류한 것 중, 실제 'True' ~> 'TP / TP + FP'

        System.out.println("Recall by label:");
        i = 0;
        double[] recLabel = trainingSummary.recallByLabel();
        for (double rec : recLabel) {
            System.out.println("label " + i + ": " + rec);
            i++;
        } // Recll(재현율): 실제 'True'인 것 중, 모델이 'True'라고 분류한 것. ~> 'TP / TP + FN'

        System.out.println("F-measure by label:");
        i = 0;
        double[] fLabel = trainingSummary.fMeasureByLabel();
        for (double f : fLabel) {
            System.out.println("label " + i + ": " + f);
            i++;
        } // F-measure: 가중치를 고려한 조화평균. 'Precision'과 'Recall'에 가중치를 적용하고 조화 평균을 계산한다.

        double accuracy = trainingSummary.accuracy();
        double falsePositiveRate = trainingSummary.weightedFalsePositiveRate();
        double truePositiveRate = trainingSummary.weightedTruePositiveRate();
        double fMeasure = trainingSummary.weightedFMeasure();
        double precision = trainingSummary.weightedPrecision();
        double recall = trainingSummary.weightedRecall();
        System.out.println("Accuracy: " + accuracy);
        System.out.println("FPR: " + falsePositiveRate);
        System.out.println("TPR: " + truePositiveRate);
        System.out.println("F-measure: " + fMeasure);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double acc = evaluator.evaluate(predicted);
        System.out.println("Accuracy on test data = " + acc);

        CrossValidator crossval = new CrossValidator().setEstimator(lr).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[] {0.01, 0.03, 0.05, 0.3, 0.5})
                .addGrid(lr.maxIter(), new int[] {100, 200, 300, 500})
                .addGrid(lr.elasticNetParam(), new double[] {0.0, 0.5, 1.0})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
        crossval.setNumFolds(5);    // K-Fold 검증

        CrossValidatorModel CV_model = crossval.fit(train_set);
        Model lr_model_best = CV_model.bestModel();

        Dataset<Row> predicted_best = lr_model_best.transform(test_set);
        predicted_best.select("target", "label", "prediction").show();

        double acc_best = evaluator.evaluate(predicted_best);
        System.out.println(acc_best);

    }
}