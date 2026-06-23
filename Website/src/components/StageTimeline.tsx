import type { WorkflowStage } from '../types/sr'

const STAGE_LABELS: Array<{ key: WorkflowStage | 'ready'; label: string }> = [
  { key: 'ready', label: 'Ready' },
  { key: 'uploaded', label: 'Image Uploaded' },
  { key: 'enhancing', label: 'Generating SR' },
  { key: 'completed', label: 'Output Ready' },
]

const stageOrder: WorkflowStage[] = ['waiting', 'uploaded', 'enhancing', 'completed']

function stageProgressIndex(stage: WorkflowStage): number {
  if (stage === 'failed') {
    return 0
  }

  return stageOrder.indexOf(stage)
}

export function StageTimeline({ stage, statusLine }: { stage: WorkflowStage; statusLine: string }) {
  const progressIndex = stageProgressIndex(stage)

  return (
    <div className="border-b border-neutral-200 bg-white bg-opacity-70 backdrop-blur-sm px-6 py-4 sm:px-8">
      <p className="text-xs font-semibold uppercase tracking-widest text-neutral-600 mb-3">Processing Pipeline</p>
      <div className="grid gap-2 sm:grid-cols-4 sm:gap-0">
        {STAGE_LABELS.map((item, idx) => {
          const active = progressIndex >= idx
          const isLast = idx === STAGE_LABELS.length - 1
          
          return (
            <div key={item.key} className="relative flex items-center">
              <div
                className={`flex-1 rounded-lg ${
                  idx === 0 ? 'rounded-r-none' : ''
                } ${isLast ? 'rounded-l-none' : ''} px-3 py-2.5 text-center transition-all ${
                  active
                    ? 'bg-gradient-to-r from-neutral-900 to-neutral-800 text-white shadow-sm'
                    : 'bg-neutral-100 text-neutral-600 border border-neutral-200'
                }`}
              >
                <p className="text-xs font-bold uppercase tracking-wider">{item.label}</p>
              </div>
              {!isLast && (
                <div className={`h-0.5 w-2 ${active ? 'bg-gradient-to-r from-neutral-900 to-neutral-700' : 'bg-neutral-200'}`} />
              )}
            </div>
          )
        })}
      </div>
      <p className="mt-3 text-xs text-neutral-600 font-medium">{statusLine}</p>
    </div>
  )
}
