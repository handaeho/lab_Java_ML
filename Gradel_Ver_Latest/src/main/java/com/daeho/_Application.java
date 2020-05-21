package com.daeho;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@SpringBootApplication
public class _Application {

	public static void main(String[] args) {
		SpringApplication.run(_Application.class, args);
	}

//	@GetMapping
//	public String welcome() {
//		return "Welcome!";
//	}

@Component
public class _Runner implements ApplicationRunner {
	@Autowired
	JdbcTemplate jdbcTemplate;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		String sql = "SELECT * FROM mlr";
		jdbcTemplate.execute(sql);
	}
}
}




