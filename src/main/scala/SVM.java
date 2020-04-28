import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

import java.util.ArrayList;
import java.util.Arrays;

public class SVM {
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
                .add("OX", "double");

        Dataset<Row> indataset = spark.read().option("header", "true").schema(schema).csv("dataset_OX.csv");

        ArrayList<String> inputColsList = new ArrayList<String>(Arrays.asList(indataset.columns()));
        System.out.println(inputColsList);

        inputColsList.remove("OX");
        System.out.println(inputColsList);

        String[] inputCols = inputColsList.parallelStream().toArray(String[]::new);
        System.out.println(inputCols);

        VectorAssembler assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features");
        Dataset<Row> dataset = assembler.transform(indataset);
        dataset.show();

        Dataset<Row>[] set = dataset.randomSplit(new double[] {0.8, 0.2}, 421);
        Dataset<Row> train_set = set[0];
        Dataset<Row> test_set = set[1];

        LinearSVC lsvc = new LinearSVC()
                .setMaxIter(100)
                .setRegParam(0.03)
                .setLabelCol("OX")
                .setFeaturesCol("features");

        LinearSVCModel lsvc_model = lsvc.fit(train_set);

        Dataset<Row> predicted = lsvc_model.transform(test_set);
        predicted.select("OX", "prediction").show();

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("OX")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC");

        double areaUnderROC = evaluator.evaluate(predicted);
        System.out.println("areaUnderROC = " + areaUnderROC);

        CrossValidator crossval = new CrossValidator().setEstimator(lsvc).setEvaluator(evaluator);
        ParamMap[] paramgrid = new ParamGridBuilder()
                .addGrid(lsvc.maxIter(), new int[] {100, 200, 300, 400, 500})
                .addGrid(lsvc.regParam(), new double[] {0.01, 0.03, 0.05, 0.3})
                .build();

        crossval.setEstimatorParamMaps(paramgrid);
        crossval.setNumFolds(5);

        CrossValidatorModel cv_model = crossval.fit(train_set);
        Model lsvm_model_best = cv_model.bestModel();

        Dataset<Row> prediction_best = lsvm_model_best.transform(test_set);
        prediction_best.select("OX", "prediction", "features").show();

        double areaUnderROC_best = evaluator.evaluate(prediction_best);
        System.out.println("areaUnderROC Best = " + areaUnderROC);
    }
}
/*
+---+----------+--------------------+
| OX|prediction|            features|
+---+----------+--------------------+
|1.0|       1.0|[0.03,0.0,0.008,0...|
|1.0|       1.0|[0.03,0.021,0.008...|
|1.0|       1.0|[0.03,0.03,0.0,0....|
|1.0|       1.0|[0.03,0.03,0.0,0....|
|1.0|       1.0|[0.03,0.03,0.019,...|
|0.0|       1.0|[0.03,0.03,0.019,...|
|0.0|       1.0|[0.03,0.03,0.02,0...|
|1.0|       1.0|[0.03,0.03,0.02,0...|
|0.0|       1.0|[0.03,0.03,0.034,...|
|0.0|       1.0|[0.03,0.03,0.0360...|
|1.0|       1.0|[0.03,0.03,0.0360...|
|0.0|       1.0|[0.03,0.03,0.0360...|
|0.0|       1.0|[0.03,0.03,0.0370...|
|1.0|       1.0|[0.03,0.03,0.0370...|
|1.0|       1.0|[0.03,0.03,0.0370...|
|1.0|       1.0|[0.03,0.03,0.0370...|
|0.0|       1.0|[0.03,0.03,0.038,...|
|0.0|       1.0|[0.03,0.03,0.038,...|
|1.0|       1.0|[0.03,0.03,0.038,...|
|0.0|       1.0|[0.03,0.03,0.038,...|
+---+----------+--------------------+
only showing top 20 rows

Accuracy Best = 0.5002520379114803

~> Spark에는 비선형 SVM이 없나? 커널 트릭 사용 방법?
 */
