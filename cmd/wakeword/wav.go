package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// wavSampleRate and the other constants describe the only WAV layout accepted: what the
// frontend consumes, 16 kHz mono 16-bit PCM.
const (
	wavSampleRate = 16000
	wavChannels   = 1
	wavBits       = 16
)

var errNotWAV = errors.New("not a RIFF/WAVE file")

// readWAV decodes a 16 kHz mono 16-bit PCM WAV stream and returns its samples. Any other
// layout is rejected with an error naming what differs. Chunks other than fmt and data are
// skipped. A data chunk with size 0 or 0xFFFFFFFF (streaming writers) extends to the end.
func readWAV(r io.Reader) ([]int16, error) {
	var header [12]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, fmt.Errorf("%w: %v", errNotWAV, err)
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, errNotWAV
	}
	haveFmt := false
	for {
		var ch [8]byte
		if _, err := io.ReadFull(r, ch[:]); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil, errors.New("wav: no data chunk")
			}
			return nil, err
		}
		id := string(ch[0:4])
		size := binary.LittleEndian.Uint32(ch[4:8])
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, fmt.Errorf("wav: fmt chunk of %d bytes", size)
			}
			buf := make([]byte, size+size%2)
			if _, err := io.ReadFull(r, buf); err != nil {
				return nil, fmt.Errorf("wav: fmt chunk: %w", err)
			}
			format := binary.LittleEndian.Uint16(buf[0:2])
			channels := binary.LittleEndian.Uint16(buf[2:4])
			rate := binary.LittleEndian.Uint32(buf[4:8])
			bits := binary.LittleEndian.Uint16(buf[14:16])
			if format == 0xFFFE && size >= 40 { // WAVE_FORMAT_EXTENSIBLE: the sub-format leads the GUID
				format = binary.LittleEndian.Uint16(buf[24:26])
			}
			if format != 1 {
				return nil, fmt.Errorf("wav: format tag %d, want 1 (PCM)", format)
			}
			if channels != wavChannels {
				return nil, fmt.Errorf("wav: %d channels, want %d", channels, wavChannels)
			}
			if rate != wavSampleRate {
				return nil, fmt.Errorf("wav: sample rate %d Hz, want %d", rate, wavSampleRate)
			}
			if bits != wavBits {
				return nil, fmt.Errorf("wav: %d bits per sample, want %d", bits, wavBits)
			}
			haveFmt = true
		case "data":
			if !haveFmt {
				return nil, errors.New("wav: data chunk before fmt chunk")
			}
			var raw []byte
			var err error
			if size == 0 || size == 0xFFFFFFFF {
				raw, err = io.ReadAll(r)
			} else {
				raw = make([]byte, size)
				_, err = io.ReadFull(r, raw)
			}
			if err != nil {
				return nil, fmt.Errorf("wav: data chunk: %w", err)
			}
			return decodePCM16(raw), nil
		default:
			if _, err := io.CopyN(io.Discard, r, int64(size+size%2)); err != nil {
				return nil, fmt.Errorf("wav: chunk %q: %w", id, err)
			}
		}
	}
}

// decodePCM16 converts little-endian 16-bit samples; a trailing odd byte is dropped.
func decodePCM16(raw []byte) []int16 {
	pcm := make([]int16, len(raw)/2)
	for i := range pcm {
		pcm[i] = int16(binary.LittleEndian.Uint16(raw[2*i:]))
	}
	return pcm
}
