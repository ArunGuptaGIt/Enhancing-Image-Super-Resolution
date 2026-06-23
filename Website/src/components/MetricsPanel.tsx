import type { QualityMetrics } from '../types/sr'

function MetricItem({ label, value }: { label: string; value: number }) {
  return (
    <div className="group rounded-xl border border-neutral-200 bg-white p-4 shadow-sm transition-all hover:shadow-elevated hover:border-neutral-300">
      <p className="text-xs font-semibold uppercase tracking-widest text-neutral-500">{label}</p>
      <p className="mt-2 text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-neutral-900 to-neutral-700">{value.toFixed(4)}</p>
    </div>
  )
}

export function MetricsPanel({ metrics }: { metrics: QualityMetrics }) {
  return (
    <div className="mt-4 animate-fade-in-up">
      <p className="mb-3 text-sm font-semibold uppercase tracking-widest text-neutral-600">Quality Metrics</p>
      <div className="grid gap-3 sm:grid-cols-3">
        <MetricItem label="PSNR (dB)" value={metrics.psnr} />
        <MetricItem label="SSIM" value={metrics.ssim} />
        <MetricItem label="LPIPS" value={metrics.lpips} />
      </div>
    </div>
  )
}
