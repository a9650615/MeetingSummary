import Foundation
import Qwen3ASR

// Persistent ANE transcriber. Loads Qwen3-ASR CoreML ONCE, then serves requests
// from stdin: [4-byte big-endian length N][N bytes of 16 kHz mono int16-LE PCM].
// Per request -> transcribe on the Neural Engine -> one JSON line {"text":...} on
// stdout. Prints "READY" to stderr once the model is loaded. The CoreML encoder is
// fixed at 30s, so the caller must send utterances <= ~29s (live utterances are).
@main
struct Qwen3Ane {
    static func main() async {
        let model: CoreMLASRModel
        do {
            model = try await CoreMLASRModel.fromPretrained()
        } catch {
            FileHandle.standardError.write(Data("ANE load failed: \(error)\n".utf8))
            exit(1)
        }
        FileHandle.standardError.write(Data("READY\n".utf8))

        let stdin = FileHandle.standardInput
        let stdout = FileHandle.standardOutput
        func readExactly(_ n: Int) -> Data? {
            var buf = Data()
            while buf.count < n {
                guard let c = try? stdin.read(upToCount: n - buf.count), !c.isEmpty else {
                    return nil
                }
                buf.append(c)
            }
            return buf
        }

        while let hdr = readExactly(4) {
            let n = Int(hdr.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).bigEndian })
            if n == 0 { continue }
            guard let body = readExactly(n) else { break }
            let samples: [Float] = body.withUnsafeBytes { raw in
                raw.bindMemory(to: Int16.self).map { Float(Int16(littleEndian: $0)) / 32768.0 }
            }
            let text = model.transcribe(audio: samples, sampleRate: 16000, language: nil)
            let data = (try? JSONSerialization.data(withJSONObject: ["text": text])) ?? Data("{}".utf8)
            stdout.write(data)
            stdout.write(Data("\n".utf8))
        }
    }
}
