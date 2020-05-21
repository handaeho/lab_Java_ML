package com.daeho;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;

import java.util.Properties;

public class Save {
    public static void main(String[] args) {
        Dataset<Row> blr_result = BLR_class.execute();
        Dataset<Row> dt_result = DT_class.execute();
        Dataset<Row> mlp_result = MLP_class.execute();
        Dataset<Row> mlr_result = MLR_class.execute();
        Dataset<Row> rfc_result = RFC_class.execute();
        Dataset<Row> rfr_result = RFR_class.execute();
        Dataset<Row> svm_result = SVM_class.execute();

        Properties properties = new Properties();
        properties.put("user", "root");
        properties.put("password", "0000");

        String url = "jdbc:mysql://127.0.0.1:3306/test?serverTimezone=UTC&useSSL=false";

//        blr_result.write().mode(SaveMode.Append).jdbc(url,"BLR", properties);
//        dt_result.write().mode(SaveMode.Append).jdbc(url,"DT", properties);
//        mlp_result.write().mode(SaveMode.Append).jdbc(url,"MLP", properties);
//        mlr_result.write().mode(SaveMode.Append).jdbc(url,"MLR", properties);
//        rfc_result.write().mode(SaveMode.Append).jdbc(url,"RFC", properties);
//        rfr_result.write().mode(SaveMode.Append).jdbc(url,"RFR", properties);
//        svm_result.write().mode(SaveMode.Append).jdbc(url,"SVM", properties);

        // 기존 테이블에 새로운 데이터 Over-write
        // (참고) 'ALTER TABLE'이 아닌 'DROP TABLE' 후, 'CREATE TABLE'를 한다.
//        blr_result.write().mode(SaveMode.Overwrite).jdbc(url,"BLR", properties);
//        dt_result.write().mode(SaveMode.Overwrite).jdbc(url,"DT", properties);
//        mlp_result.write().mode(SaveMode.Overwrite).jdbc(url,"MLP", properties);
//        mlr_result.write().mode(SaveMode.Overwrite).jdbc(url,"MLR", properties );
//        rfc_result.write().mode(SaveMode.Overwrite).jdbc(url,"RFC", properties);
//        rfr_result.write().mode(SaveMode.Overwrite).jdbc(url,"RFR", properties);
        svm_result.write().mode(SaveMode.Overwrite).jdbc(url,"SVM", properties);
    }
}
