package com.daeho;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;

@Repository
public class ML_database {
    @Autowired
    JdbcTemplate jdbcTemplate;

    public List<String> get_table() {
        List<String> table = new ArrayList<>();

        table.addAll(jdbcTemplate.queryForList("select * from blr", String.class));
        return table;
    }
}
