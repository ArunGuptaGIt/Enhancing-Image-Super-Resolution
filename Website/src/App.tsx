import { useCallback, useEffect, useMemo, useState } from 'react'
import { enhanceImage, fetchHealth, selectModel } from './api/enhanceApi'
import { ComparisonViewer } from './components/ComparisonViewer'
import { ControlsPanel } from './components/ControlsPanel'
import { StageTimeline } from './components/StageTimeline'
import type { EnhanceResponse, HealthResponse, HistoryItem, WorkflowStage } from './types/sr'

type ComparisonMode = 'single' | 'slider' | 'sideBySide'

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewSrc, setPreviewSrc] = useState<string | null>(null)
  const [uploadedResolution, setUploadedResolution] = useState<[number, number] | null>(null)
  const [result, setResult] = useState<EnhanceResponse | null>(null)
  const [stage, setStage] = useState<WorkflowStage>('waiting')
  const [sliderValue, setSliderValue] = useState(50)
  const [showMetrics, setShowMetrics] = useState(true)
  const [isRunning, setIsRunning] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>('single')
  const [liveElapsedMs, setLiveElapsedMs] = useState(0)
  const [generationStartedAt, setGenerationStartedAt] = useState<number | null>(null)
  const [isSwitchingModel, setIsSwitchingModel] = useState(false)

  const loadHealth = useCallback(async () => {
    try {
      const body = await fetchHealth()
      setHealth(body)
      setHealthError(null)
      if (body.model_status !== 'loaded') {
        setErrorMessage(body.model.error ?? 'Model not loaded')
      }
    } catch (error) {
      setHealthError(error instanceof Error ? error.message : 'Failed to load health status.')
    }
  }, [])

  useEffect(() => {
    let mounted = true

    const loadHealthOnMount = async () => {
      try {
        const body = await fetchHealth()
        if (!mounted) {
          return
        }

        setHealth(body)
        setHealthError(null)
        if (body.model_status !== 'loaded') {
          setErrorMessage(body.model.error ?? 'Model not loaded')
        }
      } catch (error) {
        if (!mounted) {
          return
        }
        setHealthError(error instanceof Error ? error.message : 'Failed to load health status.')
      }
    }

    void loadHealthOnMount()

    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    if (!isRunning || generationStartedAt === null) {
      return
    }

    const timer = window.setInterval(() => {
      setLiveElapsedMs(Date.now() - generationStartedAt)
    }, 100)

    return () => {
      window.clearInterval(timer)
    }
  }, [isRunning, generationStartedAt])

  const statusLine = useMemo(() => {
    if (stage === 'failed') {
      return 'Pipeline interrupted. Please try another image or retry generation.'
    }

    if (stage === 'completed' && result) {
      const completedSeconds = liveElapsedMs > 0 ? liveElapsedMs / 1000 : result.timings_ms.total / 1000
      return `Completed in ${completedSeconds.toFixed(2)} s` 
      // with ${result.model.backend}.`
    }

    if (stage === 'enhancing') {
      return `Generating... ${(liveElapsedMs / 1000).toFixed(1)} s`
    }

    if (healthError) {
      return health ? 'Backend health check retrying...' : 'Backend health check unavailable.'
    }

    if (!health) {
      return 'Checking model status...'
    }

    if (health.model_status !== 'loaded') {
      return 'Model not loaded'
    }

    const loadSeconds = health.model.load_time_ms ? (health.model.load_time_ms / 1000).toFixed(2) : null
    return loadSeconds ? `Model loaded in ${loadSeconds} s` : 'Model loaded'
  }, [result, stage, liveElapsedMs, health, healthError])

  const handleUpload = (file: File | null) => {
    if (!file) {
      return
    }

    setSelectedFile(file)
    const objectUrl = URL.createObjectURL(file)
    setPreviewSrc(objectUrl)
    const image = new Image()
    image.onload = () => {
      setUploadedResolution([image.width, image.height])
    }
    image.src = objectUrl
    setResult(null)
    setErrorMessage(null)
    setHealthError(null)
    setStage('uploaded')
  }

  const runEnhancement = async () => {
    if (!selectedFile) {
      setErrorMessage('Please upload an image first.')
      return
    }

    if (isSwitchingModel) {
      return
    }

    setErrorMessage(null)
    setIsRunning(true)
    setStage('enhancing')
    const startedAt = Date.now()
    setGenerationStartedAt(startedAt)
    setLiveElapsedMs(0)

    try {
      const body = await enhanceImage(selectedFile)
      const frontendElapsedMs = Date.now() - startedAt
      setResult(body)
      setStage('completed')

      const historyEntry: HistoryItem = {
        id: body.request_id,
        runtime: frontendElapsedMs / 1000,
        modelName: body.model.backend,
        outputSize: `${body.dimensions.sr[0]}x${body.dimensions.sr[1]}`,
      }

      setHistory((previous) => [historyEntry, ...previous].slice(0, 6))
    } catch (error) {
      setStage('failed')
      setErrorMessage(error instanceof Error ? error.message : 'Unknown error during enhancement.')
    } finally {
      setIsRunning(false)
      setLiveElapsedMs(Date.now() - startedAt)
      setGenerationStartedAt(null)
    }
  }

  const handleModelChange = async (modelId: string) => {
    if (!health || modelId === health.model.selected_id) {
      return
    }

    setErrorMessage(null)
    setHealthError(null)
    setIsSwitchingModel(true)

    try {
      const body = await selectModel(modelId)
      setHealth(body)
      if (body.model_status !== 'loaded') {
        setErrorMessage(body.model.error ?? 'Model not loaded')
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to switch model.')
    } finally {
      setIsSwitchingModel(false)
    }
  }

  const downloadResult = () => {
    if (!result?.images.sr) {
      return
    }

    const link = document.createElement('a')
    link.href = result.images.sr
    link.download = `sr-output-${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const resetSession = () => {
    setSelectedFile(null)
    setPreviewSrc(null)
    setUploadedResolution(null)
    setResult(null)
    setErrorMessage(null)
    setHealthError(null)
    setStage('waiting')
    setSliderValue(50)
    setHistory([])
    setLiveElapsedMs(0)
    setGenerationStartedAt(null)
    void loadHealth()
  }

  return (
    <main className="mx-auto flex h-screen max-w-7xl flex-col overflow-hidden bg-gradient-to-br from-white via-neutral-50 to-neutral-100">
      <header className="border-b border-neutral-200 bg-white bg-opacity-70 backdrop-blur-md px-5 py-2.5 sm:px-7 sm:py-3 animate-fade-in-up">
        <div className="flex items-center justify-between">
          <div>
            
            <h1 className="mt-3 text-2xl sm:text-3xl lg:text-4xl font-black tracking-tight break-words">
              <span className="sr-clarity-title" data-text="Enhancing Image Super Resolution">
                <span className="sr-clarity-base">Enhancing Image Super Resolution</span>
              </span>
            </h1>
          </div>
        </div>
      </header>

      <StageTimeline stage={stage} statusLine={statusLine} />

      <section className="flex flex-1 min-h-0 gap-3 overflow-hidden px-3 py-2 sm:px-5 sm:py-3 lg:px-6 lg:py-4 flex-col lg:flex-row">
        <div className="w-full h-full min-h-0 space-y-3 overflow-y-auto flex flex-col pr-1 lg:w-80 lg:flex-shrink-0 lg:min-h-[620px]">
          <ControlsPanel
            fileName={selectedFile?.name ?? null}
            previewSrc={previewSrc}
            isRunning={isRunning}
            isSwitchingModel={isSwitchingModel}
            modelOptions={health?.model.options ?? []}
            selectedModelId={health?.model.selected_id ?? null}
            onModelChange={handleModelChange}
            onFileSelect={handleUpload}
            onGenerate={runEnhancement}
            onReset={resetSession}
          />

          <section className="rounded-2xl border border-neutral-200 bg-white shadow-elevated p-3 animate-fade-in">
            <h3 className="text-xs font-bold uppercase tracking-widest text-neutral-700 mb-2">Image Details</h3>
            <div className="space-y-1.5 text-xs text-neutral-700">
              <div className="flex items-center justify-between">
                <span className="font-medium">Input Resolution</span>
                <span className="font-semibold text-neutral-900">{uploadedResolution ? `${uploadedResolution[0]}×${uploadedResolution[1]} px` : '—'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="font-medium">SR Resolution</span>
                <span className="font-semibold text-neutral-900">{result?.dimensions.sr ? `${result.dimensions.sr[0]}×${result.dimensions.sr[1]} px` : '—'}</span>
              </div>
              {showMetrics && (
                <div className="mt-2 border-t border-neutral-200 pt-2 space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">PSNR</span>
                    <span className="font-semibold text-neutral-900">{result?.quality ? `${result.quality.psnr.toFixed(1)} dB` : '—'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="font-medium">SSIM</span>
                    <span className="font-semibold text-neutral-900">{result?.quality ? result.quality.ssim.toFixed(3) : '—'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="font-medium">LPIPS</span>
                    <span className="font-semibold text-neutral-900">{result?.quality ? result.quality.lpips.toFixed(3) : '—'}</span>
                  </div>
                </div>
              )}
            </div>
          </section>

          {errorMessage && (
            <div className="animate-slide-in rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700">
              {errorMessage}
            </div>
          )}

          {healthError && (
            <div className="animate-slide-in rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-medium text-amber-700">
              {healthError}
            </div>
          )}

          {!!history.length && (
            <section className="rounded-2xl border border-neutral-200 bg-white shadow-elevated p-3 animate-fade-in">
              <h3 className="text-xs font-bold uppercase tracking-widest text-neutral-700 mb-3">Recent Runs</h3>
              <div className={`space-y-2 ${history.length > 1 ? 'max-h-16 overflow-y-auto' : ''}`}>
                {history.slice(0, 6).map((item) => (
                  <div key={item.id} className="rounded-lg border border-neutral-200 bg-gradient-to-r from-neutral-50 to-white hover:from-neutral-100 hover:to-neutral-50 px-3 py-2 text-xs transition-all">
                    <p className="font-semibold text-neutral-900">{item.modelName}</p>
                    <p className="text-neutral-600 mt-1">{item.outputSize} • {item.runtime.toFixed(2)} s</p>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>

        <div className="min-w-0 h-full overflow-y-auto lg:flex-1">
          <ComparisonViewer
            lrSrc={previewSrc}
            hrSrc={result?.images.sr ?? null}
            sliderValue={sliderValue}
            onSliderChange={setSliderValue}
            comparisonMode={comparisonMode}
            onSetComparisonMode={setComparisonMode}
            onDownload={downloadResult}
            showMetrics={showMetrics}
            onToggleMetrics={() => setShowMetrics((previous) => !previous)}
          />
        </div>
      </section>
    </main>
  )
}

export default App
