package com.sim.platform.controller;

import com.sim.platform.common.Result;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * 健康检查控制器
 * Health Check Controller
 */
@Tag(name = "Health Check", description = "健康检查接口")
@RestController
@RequestMapping("/api/health")
public class HealthController {

    @Operation(summary = "Health Check", description = "检查服务健康状态")
    @GetMapping
    public Result<Map<String, Object>> health() {
        Map<String, Object> health = new HashMap<>();
        health.put("status", "UP");
        health.put("service", "Intelligent Simulation Platform");
        health.put("version", "1.0.0");
        health.put("timestamp", LocalDateTime.now());
        return Result.success(health);
    }
}
