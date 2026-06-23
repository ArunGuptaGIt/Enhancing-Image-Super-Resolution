import type { ModelOption } from '../types/sr'

interface ControlsPanelProps {
  fileName: string | null
  previewSrc: string | null
  isRunning: boolean
  isSwitchingModel: boolean
  modelOptions: ModelOption[]
  selectedModelId: string | null
  onModelChange: (modelId: string) => void
  onFileSelect: (file: File | null) => void
  onGenerate: () => void
  onReset: () => void
}

export function ControlsPanel({
  fileName,
  previewSrc,
  isRunning,
  isSwitchingModel,
  modelOptions,
  selectedModelId,
  onModelChange,
  onFileSelect,
  onGenerate,
  onReset,
}: ControlsPanelProps) {
  const hasMultipleModels = modelOptions.length > 1

  return (
    <aside className="space-y-3">
      <div className="rounded-2xl border border-neutral-200 bg-white shadow-elevated p-4 animate-fade-in-up">
        <h2 className="text-sm font-bold uppercase tracking-widest text-neutral-900">Upload Image</h2>
        <p className="mt-1.5 text-xs leading-relaxed text-neutral-600">Select a low-resolution image for 4x enhancement</p>

        <label className="group mt-3 flex min-h-28 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-neutral-300 bg-gradient-to-br from-neutral-50 to-white px-4 py-3 text-center transition-all hover:border-neutral-400 hover:bg-neutral-50 overflow-hidden">
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(event) => onFileSelect(event.target.files?.[0] ?? null)}
          />
          {previewSrc ? (
            <div className="w-full">
              <img
                src={previewSrc}
                alt="Uploaded LR preview"
                className="h-24 w-full rounded-md border border-neutral-200 object-cover"
              />
              <span className="mt-2 block text-xs font-semibold text-neutral-900">Change image</span>
            </div>
          ) : (
            <>
              <svg className="h-8 w-8 text-neutral-400 group-hover:text-neutral-600 transition" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              <span className="mt-2 text-sm font-semibold text-neutral-900">Click to upload</span>
            </>
          )}
          {/* <span className="mt-1 text-xs text-neutral-500">PNG, JPG, WEBP</span> */}
        </label>

        {fileName && (
          <div className="mt-3 animate-slide-in flex items-center gap-2 rounded-lg border border-neutral-200 bg-gradient-to-r from-green-50 to-emerald-50 px-3 py-2">
            <svg className="h-4 w-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            <p className="truncate text-xs font-medium text-neutral-900">{fileName}</p>
          </div>
        )}

        {hasMultipleModels && (
          <div className="mt-3 space-y-1.5">
            <p className="text-xs font-semibold uppercase tracking-widest text-neutral-700">Model</p>
            <select
              value={selectedModelId ?? ''}
              onChange={(event) => onModelChange(event.target.value)}
              disabled={isSwitchingModel || isRunning}
              className="w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 text-sm text-neutral-900 shadow-sm disabled:cursor-not-allowed disabled:bg-neutral-100 disabled:text-neutral-500"
            >
              {modelOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
        <button
          type="button"
          onClick={onGenerate}
          disabled={isRunning || isSwitchingModel || !fileName}
          className="group relative overflow-hidden rounded-lg bg-gradient-to-r from-neutral-900 to-neutral-800 px-4 py-2.5 text-sm font-bold text-white shadow-elevated transition-all hover:shadow-hover disabled:cursor-not-allowed disabled:from-neutral-400 disabled:to-neutral-300 disabled:shadow-none"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/10 to-white/0 opacity-0 group-hover:opacity-100 transition-opacity" />
          <span className="relative">{isSwitchingModel ? 'Switching model...' : isRunning ? 'Generating...' : 'Generate 4x'}</span>
        </button>
        <button
          type="button"
          onClick={onReset}
          className="group relative rounded-lg border-2 border-neutral-300 bg-white px-4 py-2.5 text-sm font-bold text-neutral-900 transition-all hover:bg-neutral-50 hover:border-neutral-400 shadow-sm"
        >
          Reset
        </button>
      </div>
    </aside>
  )
}
