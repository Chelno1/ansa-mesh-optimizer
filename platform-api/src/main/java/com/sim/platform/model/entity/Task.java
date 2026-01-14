package com.sim.platform.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 任务实体
 * Task Entity
 */
@Data
@TableName("task")
public class Task {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private Long workflowInstanceId;
    
    private String name;
    
    private String type;
    
    private Integer stepIndex;
    
    private String status;
    
    private Integer priority;
    
    private String params;
    
    private String result;
    
    private String errorMessage;
    
    private Integer retryCount;
    
    private String agentId;
    
    private LocalDateTime startedAt;
    
    private LocalDateTime completedAt;
    
    private LocalDateTime createdAt;
}
