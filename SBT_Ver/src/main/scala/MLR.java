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

        Dataset<Row> indataset = spark.read().option("header", "true").schema(schema).csv("dataset.csv");

        ArrayList<String> inputColsList = new ArrayList<String>(Arrays.asList(indataset.columns()));
        System.out.println(inputColsList);

        //inputColsList.remove("target");
        inputColsList.remove("label");
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

        System.out.println(predicted_best.toJavaRDD().first());

    }
}
/*
+------+-----+----------+
|target|label|prediction|
+------+-----+----------+
|  37.2|  1.0|       1.0|
|  10.8|  0.0|       0.0|
|  61.0|  3.0|       3.0|
|  89.1|  4.0|       4.0|
|   3.3|  0.0|       0.0|
|  31.7|  1.0|       1.0|
|  53.4|  2.0|       2.0|
|  74.7|  3.0|       3.0|
|  72.1|  3.0|       3.0|
|   0.1|  0.0|       0.0|
|  87.7|  4.0|       4.0|
|  97.1|  4.0|       4.0|
|  90.9|  4.0|       4.0|
|  82.1|  4.0|       4.0|
|  57.3|  2.0|       2.0|
|  22.2|  1.0|       0.0|
|  72.5|  3.0|       3.0|
|   4.6|  0.0|       0.0|
|  77.5|  3.0|       4.0|
|  99.7|  4.0|       4.0|
+------+-----+----------+
only showing top 20 rows

0.9097455592894863

~>  임의로 target과 label을 생성했는데 정확도가 지나치게 높은 느낌.
    다른 요소에 비해 target 값이 매우 커서 label 예측/분류에도 영향을 크게 주는 것으로 사료됨.
    (target 범위 따라 나눈 label 따라서 그냥 가는 느낌)

>> setWeightCol("target") 설정 시,
+------+-----+----------+
|target|label|prediction|
+------+-----+----------+
|  37.2|  1.0|       1.0|
|  10.8|  0.0|       1.0|
|  61.0|  3.0|       3.0|
|  89.1|  4.0|       4.0|
|   3.3|  0.0|       1.0|
|  31.7|  1.0|       1.0|
|  53.4|  2.0|       2.0|
|  74.7|  3.0|       3.0|
|  72.1|  3.0|       3.0|
|   0.1|  0.0|       1.0|
|  87.7|  4.0|       4.0|
|  97.1|  4.0|       4.0|
|  90.9|  4.0|       4.0|
|  82.1|  4.0|       4.0|
|  57.3|  2.0|       3.0|
|  22.2|  1.0|       1.0|
|  72.5|  3.0|       3.0|
|   4.6|  0.0|       1.0|
|  77.5|  3.0|       4.0|
|  99.7|  4.0|       4.0|
+------+-----+----------+
only showing top 20 rows

0.7004320691310609
~>  target 컬럼에 가중치를 부여함. 정확도가 조금 감소한 것으로 보면 가중치가 적용된 것 같기는 함.
    가중치를 부여하는 컬럼을 바꾸어보며 테스트?

elasticNetParam Setting (.addGrid(lr.elasticNetParam(), new double[] {0.0, 0.5, 1.0}))
 ~> 0 ~ 1 사이의 값으로, 0에 가까울수록 L1 / 1에 가까울수록 L2 정규화 규제를 설정한다.
L1 규제: 계수(가중치)의 절대값에 비례하는 페널티 추가. -> Lasso Regression. 일부 계수 값을 0으로 만들수 있다.
L2 규제: 계수(가중치)의 제곱에 비례하는 페널티 추가. -> Ridge Regression. 계수 값을 최소화 할 수는 있지만 0으로 만들수는 없다.
+------+-----+----------+
|target|label|prediction|
+------+-----+----------+
|  37.2|  1.0|       1.0|
|  10.8|  0.0|       0.0|
|  61.0|  3.0|       3.0|
|  89.1|  4.0|       4.0|
|   3.3|  0.0|       0.0|
|  31.7|  1.0|       1.0|
|  53.4|  2.0|       2.0|
|  74.7|  3.0|       3.0|
|  72.1|  3.0|       3.0|
|   0.1|  0.0|       0.0|
|  87.7|  4.0|       4.0|
|  97.1|  4.0|       4.0|
|  90.9|  4.0|       4.0|
|  82.1|  4.0|       4.0|
|  57.3|  2.0|       2.0|
|  22.2|  1.0|       1.0|
|  72.5|  3.0|       3.0|
|   4.6|  0.0|       0.0|
|  77.5|  3.0|       3.0|
|  99.7|  4.0|       4.0|
+------+-----+----------+
only showing top 20 rows

0.9380700912145943
~> 엘라스틱넷(ElasticNet) : L2, L1 규제를 함께 결합한 모델.
주로 피처가 많은 데이터에서 적용되며, L1 규제로 피처의 개수를 줄임과 동시에 L2 규제로 계수 값의 크기를 조정
라쏘 회귀가 서로 상관 관계가 높은 피처들의 경우에 이들 중에서 중요 피처만을 셀렉션하고
다른 피처들은 모두 회귀 계수를 0으로 만드는 성향이 강함.
이러한 성향으로 인해 alpha 값에 따라 회귀 계수의 값이 급격히 변동
엘라스틱 넷 회귀는 이를 완화하기 위해 L2 규제를 라쏘 회귀에 추가
*/