import type { EnhanceResponse, HealthResponse } from '../types/sr'

function extractErrorMessage(payload: unknown): string | null {
  if (typeof payload === 'string' && payload.trim().length > 0) {
    return payload
  }

  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = (payload as { detail: unknown }).detail
    if (typeof detail === 'string' && detail.trim().length > 0) {
      return detail
    }
    if (detail && typeof detail === 'object' && 'message' in detail) {
      const message = (detail as { message: unknown }).message
      if (typeof message === 'string' && message.trim().length > 0) {
        return message
      }
    }
  }

  return null
}

export async function enhanceImage(file: File): Promise<EnhanceResponse> {
  const payload = new FormData()
  payload.append('file', file)

  const candidates = ['http://127.0.0.1:8000/api/enhance', '/api/enhance']
  let lastError: Error | null = null

  for (const url of candidates) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        body: payload,
      })

      if (!response.ok) {
        let message = `Enhancement failed (${response.status}).`
        try {
          const body = (await response.json()) as unknown
          message = extractErrorMessage(body) ?? message
        } catch {
          // keep parsed fallback message
        }
        lastError = new Error(message)
        continue
      }

      return (await response.json()) as EnhanceResponse
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Enhancement failed in backend.')
    }
  }

  throw lastError ?? new Error('Enhancement failed in backend.')
}

export async function fetchHealth(): Promise<HealthResponse> {
  const candidates = ['/api/health', 'http://127.0.0.1:8000/api/health']
  let lastError: Error | null = null

  for (const url of candidates) {
    try {
      const response = await fetch(url, {
        method: 'GET',
      })

      if (!response.ok) {
        lastError = new Error(`Health check failed (${response.status}) at ${url}.`)
        continue
      }

      const contentType = (response.headers.get('content-type') ?? '').toLowerCase()
      if (!contentType.includes('application/json')) {
        lastError = new Error(`Health endpoint returned non-JSON response at ${url}.`)
        continue
      }

      return (await response.json()) as HealthResponse
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Failed to fetch backend health status.')
    }
  }

  throw lastError ?? new Error('Failed to fetch backend health status.')
}

export async function selectModel(modelId: string): Promise<HealthResponse> {
  const candidates = ['/api/models/select', 'http://127.0.0.1:8000/api/models/select']
  let lastError: Error | null = null

  for (const url of candidates) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_id: modelId }),
      })

      if (!response.ok) {
        let message = `Model selection failed (${response.status}).`
        try {
          const body = (await response.json()) as unknown
          message = extractErrorMessage(body) ?? message
        } catch {
          // keep parsed fallback message
        }
        lastError = new Error(message)
        continue
      }

      return (await response.json()) as HealthResponse
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Failed to switch model in backend.')
    }
  }

  throw lastError ?? new Error('Failed to switch model in backend.')
}
