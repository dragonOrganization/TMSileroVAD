Pod::Spec.new do |s|
  s.name             = 'TMSileroVAD'
  s.version          = '0.1.0'
  s.summary          = 'Silero VAD CoreML wrapper for iOS real-time voice activity detection.'
  s.description      = <<-DESC
    A Swift Pod wrapping the FluidInference Silero VAD unified CoreML model.
    Supports two variants: 256 ms balanced (endpointing/ASR) and 32 ms realtime
    (low-latency barge-in). Built-in AVAudioConverter resampling. iOS 14+.
  DESC
  s.homepage         = 'https://github.com/dragonOrganization/TMSileroVAD'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'TalkMe' => 'ios@example.com' }
  s.source           = { :git => 'https://github.com/dragonOrganization/TMSileroVAD.git', :tag => "v#{s.version}" }
  s.ios.deployment_target = '14.0'
  s.swift_versions = ['5.9']
  s.source_files = 'Sources/TMSileroVAD/**/*.swift'
  s.resource_bundles = {
    'TMSileroVADResources' => [
      'Sources/TMSileroVAD/Resources/silero-vad-unified-v6.0.0.mlmodelc',
      'Sources/TMSileroVAD/Resources/silero-vad-unified-256ms-v6.0.0.mlmodelc'
    ]
  }
  s.frameworks = 'Foundation', 'AVFoundation', 'CoreML', 'Accelerate'
end
