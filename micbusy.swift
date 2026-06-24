// micbusy — prints "1" if the default input device (mic) is in use by ANY app,
// else "0". CoreAudio kAudioDevicePropertyDeviceIsRunningSomewhere — the signal
// Granola/Notion-style tools use; catches browser meetings + native apps + any
// mic-open, not just known processes.
//   build:  swiftc micbusy.swift -o micbusy -framework CoreAudio
import CoreAudio
import Foundation

func addr(_ sel: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel,
                               mScope: kAudioObjectPropertyScopeGlobal,
                               mElement: kAudioObjectPropertyElementMain)
}

func defaultInput() -> AudioDeviceID {
    var dev = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    var a = addr(kAudioHardwarePropertyDefaultInputDevice)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size, &dev)
    return dev
}

func micInUse(_ dev: AudioDeviceID) -> Bool {
    var running = UInt32(0)
    var size = UInt32(MemoryLayout<UInt32>.size)
    var a = addr(kAudioDevicePropertyDeviceIsRunningSomewhere)
    let st = AudioObjectGetPropertyData(dev, &a, 0, nil, &size, &running)
    return st == noErr && running != 0
}

print(micInUse(defaultInput()) ? "1" : "0")
