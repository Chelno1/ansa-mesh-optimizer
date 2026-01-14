package com.sim.platform;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * 智能仿真平台主应用
 * Intelligent Simulation Platform Main Application
 * 
 * @author Platform Team
 * @version 1.0.0
 */
@SpringBootApplication
@EnableScheduling
@MapperScan("com.sim.platform.repository")
public class PlatformApplication {

    public static void main(String[] args) {
        SpringApplication.run(PlatformApplication.class, args);
        System.out.println("========================================");
        System.out.println("  智能仿真平台启动成功!");
        System.out.println("  Platform started successfully!");
        System.out.println("  API文档: http://localhost:8080/doc.html");
        System.out.println("========================================");
    }
}
