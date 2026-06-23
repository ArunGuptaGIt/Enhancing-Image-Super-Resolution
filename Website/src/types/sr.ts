export type WorkflowStage = 'waiting' | 'uploaded' | 'enhancing' | 'completed' | 'failed'

export interface QualityMetrics {
  psnr: number
  ssim: number
  lpips: number
}

export interface EnhanceResponse {
  request_id: string
  model: {
    backend: string
    loaded_at: string
  }
  timings_ms: {
    total: number
    upscale: number
  }
  images: {
    lr: string
    sr: string
  }
  dimensions: {
    lr: [number, number]
    sr: [number, number]
  }
  quality: QualityMetrics
}

export interface HealthResponse {
  status: string
  model_status: 'loaded' | 'model not loaded'
  model: {
    backend: string
    path: string | null
    load_time_ms: number | null
    loaded_at: string
    error: string | null
    selected_id: string | null
    options: ModelOption[]
  }
}

export interface ModelOption {
  id: string
  label: string
  path: string
}

export interface HistoryItem {
  id: string
  runtime: number
  modelName: string
  outputSize: string
}
