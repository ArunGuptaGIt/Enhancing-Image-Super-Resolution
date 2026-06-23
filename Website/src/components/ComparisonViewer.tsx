import { useMemo, useState } from 'react'

interface ComparisonViewerProps {
  lrSrc: string | null
  hrSrc: string | null
  sliderValue: number
  onSliderChange: (value: number) => void
  comparisonMode: 'single' | 'slider' | 'sideBySide'
  onSetComparisonMode: (mode: 'single' | 'slider' | 'sideBySide') => void
  onDownload: () => void
  showMetrics: boolean
  onToggleMetrics: () => void
}

export function ComparisonViewer({
  lrSrc,
  hrSrc,
  sliderValue,
  onSliderChange,
  comparisonMode,
  onSetComparisonMode,
  onDownload,
  showMetrics,
  onToggleMetrics,
}: ComparisonViewerProps) {
  const hasOutput = useMemo(() => Boolean(hrSrc), [hrSrc])
  const [isDraggingDivider, setIsDraggingDivider] = useState(false)

  const updateSliderFromClientX = (clientX: number, container: HTMLDivElement | null) => {
    if (!container) {
      return
    }

    const rect = container.getBoundingClientRect()
    if (rect.width <= 0) {
      return
    }

    const raw = ((clientX - rect.left) / rect.width) * 100
    const clamped = Math.max(0, Math.min(100, raw))
    onSliderChange(clamped)
  }

  return (
    <section className="rounded-2xl border border-neutral-200 bg-white shadow-elevated flex flex-col p-4 animate-fade-in-up">
      <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="text-lg font-bold text-neutral-900 uppercase tracking-tight">Comparison</h2>

        <div className="flex flex-row flex-wrap gap-2 items-center">
          <div className="flex items-center gap-1 rounded-lg border border-neutral-300 bg-neutral-50 p-1">
            <button
              type="button"
              onClick={onDownload}
              disabled={!hasOutput}
              className="group relative overflow-hidden rounded-md bg-gradient-to-r from-neutral-900 to-neutral-800 px-3 py-1.5 text-xs font-bold text-white shadow-sm transition-all hover:shadow-md disabled:cursor-not-allowed disabled:from-neutral-400 disabled:to-neutral-300"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/10 to-white/0 opacity-0 group-hover:opacity-100 transition-opacity" />
              <span className="relative flex items-center gap-1">
                <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download
              </span>
            </button>
          </div>

          <button
            type="button"
            onClick={onToggleMetrics}
            className={`rounded-md px-3 py-1.5 text-xs font-bold transition-all whitespace-nowrap ${
              showMetrics
                ? 'bg-gradient-to-r from-neutral-900 to-neutral-800 text-white shadow-sm'
                : 'bg-white text-neutral-700 hover:bg-neutral-100 border border-neutral-300'
            }`}
          >
            Metrics
          </button>

          <div className="flex items-center gap-1 rounded-lg border border-neutral-300 bg-neutral-50 p-1">
            <button
              type="button"
              onClick={() => onSetComparisonMode('single')}
              className={`rounded-md px-3 py-1.5 text-xs font-bold transition-all whitespace-nowrap ${
                comparisonMode === 'single'
                  ? 'bg-gradient-to-r from-neutral-900 to-neutral-800 text-white shadow-sm'
                  : 'bg-white text-neutral-700 hover:bg-neutral-100'
              }`}
            >
              Single
            </button>
            <button
              type="button"
              onClick={() => onSetComparisonMode('slider')}
              className={`rounded-md px-3 py-1.5 text-xs font-bold transition-all whitespace-nowrap ${
                comparisonMode === 'slider'
                  ? 'bg-gradient-to-r from-neutral-900 to-neutral-800 text-white shadow-sm'
                  : 'bg-white text-neutral-700 hover:bg-neutral-100'
              }`}
            >
              Slider
            </button>
            <button
              type="button"
              onClick={() => onSetComparisonMode('sideBySide')}
              className={`rounded-md px-3 py-1.5 text-xs font-bold transition-all ${
                comparisonMode === 'sideBySide'
                  ? 'bg-gradient-to-r from-neutral-900 to-neutral-800 text-white shadow-sm'
                  : 'bg-white text-neutral-700 hover:bg-neutral-100'
              }`}
            >
              Side by Side
            </button>
          </div>
        </div>
      </div>

      {!lrSrc && (
        <div className="flex flex-1 min-h-96 items-center justify-center rounded-xl border-2 border-dashed border-neutral-300 bg-gradient-to-br from-neutral-50 to-white text-sm text-neutral-500">
          <div className="text-center">
            <svg className="mx-auto h-16 w-16 text-neutral-300 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="font-semibold text-neutral-900 text-base">Upload an image to start</p>
            <p className="mt-1 text-xs text-neutral-500">Your low-resolution image will appear here</p>
          </div>
        </div>
      )}

      {lrSrc && comparisonMode === 'single' && (
        <div className="space-y-2">
          <div className="relative overflow-hidden rounded-xl border border-neutral-300 bg-neutral-100 shadow-sm">
            <div className="relative h-80 sm:h-[440px] lg:h-[560px]">
              <img src={hrSrc ?? lrSrc} alt="Resolved" className="absolute inset-0 h-full w-full object-contain" />
            </div>
          </div>
        </div>
      )}

      {lrSrc && comparisonMode === 'slider' && (
        <div className="space-y-2">
          <div
            className="relative overflow-hidden rounded-xl border border-neutral-300 bg-neutral-100 shadow-sm"
            onMouseMove={(event) => {
              if (isDraggingDivider) {
                updateSliderFromClientX(event.clientX, event.currentTarget)
              }
            }}
            onMouseUp={() => setIsDraggingDivider(false)}
            onMouseLeave={() => setIsDraggingDivider(false)}
            onTouchMove={(event) => {
              if (!isDraggingDivider) {
                return
              }
              updateSliderFromClientX(event.touches[0].clientX, event.currentTarget)
            }}
            onTouchEnd={() => setIsDraggingDivider(false)}
          >
            <div className="relative h-80 sm:h-[440px] lg:h-[560px]">
              <img src={hrSrc ?? lrSrc} alt="HR" className="absolute inset-0 h-full w-full object-contain" />

              {hrSrc && (
                <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - sliderValue}% 0 0)` }}>
                  <img src={lrSrc} alt="LR overlay" className="absolute inset-0 h-full w-full object-contain" />
                </div>
              )}

              {hrSrc && (
                <>
                  <div
                    className="pointer-events-none absolute inset-y-0 w-1 bg-gradient-to-r from-neutral-900 via-white to-neutral-900 shadow-lg"
                    style={{ left: `${sliderValue}%` }}
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 z-20 w-6 -translate-x-1/2 cursor-ew-resize bg-transparent"
                    style={{ left: `${sliderValue}%` }}
                    onMouseDown={(event) => {
                      setIsDraggingDivider(true)
                      updateSliderFromClientX(event.clientX, event.currentTarget.parentElement as HTMLDivElement | null)
                    }}
                    onTouchStart={(event) => {
                      setIsDraggingDivider(true)
                      updateSliderFromClientX(event.touches[0].clientX, event.currentTarget.parentElement as HTMLDivElement | null)
                    }}
                    aria-label="Drag comparison divider"
                  >
                    <span className="pointer-events-none absolute inset-y-0 left-1/2 w-1 -translate-x-1/2 rounded bg-white/80 shadow" />
                  </button>
                </>
              )}
            </div>
          </div>

        </div>
      )}

      {lrSrc && comparisonMode === 'sideBySide' && (
        <div className="space-y-2">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col rounded-xl border border-neutral-300 overflow-hidden shadow-sm hover:shadow-elevated transition-all">
              <img src={lrSrc} alt="LR" className="h-72 w-full object-contain bg-neutral-100 sm:h-[440px] lg:h-[560px]" />
            </div>

            <div className="flex flex-col rounded-xl border border-neutral-300 overflow-hidden shadow-sm hover:shadow-elevated transition-all">
              {hrSrc ? (
                <>
                  <img src={hrSrc} alt="SR" className="h-72 w-full object-contain bg-neutral-100 sm:h-[440px] lg:h-[560px]" />
                </>
              ) : (
                <div className="flex h-72 items-center justify-center bg-gradient-to-br from-neutral-50 to-neutral-100 text-xs font-medium text-neutral-500 sm:h-[440px] lg:h-[560px]">
                  Generate 4x to see result
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
