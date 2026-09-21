// Package wav reads and writes the only audio layout the detector consumes: 16 kHz mono
// 16-bit PCM WAV. Anything else is rejected with an error naming what differs.
package wav

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

// SampleRate, Channels and Bits describe the accepted layout.
const (
	SampleRate = 16000
	Channels   = 1
	Bits       = 16
)

// ErrNotWAV reports a stream without a RIFF/WAVE header.
var ErrNotWAV = errors.New("not a RIFF/WAVE file")

// ReadFile decodes a WAV file with Read.
func ReadFile(path string) ([]int16, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	pcm, err := Read(bufio.NewReader(f))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return pcm, nil
}

// Read decodes a 16 kHz mono 16-bit PCM WAV stream and returns its samples. Chunks other
// than fmt and data are skipped. A data chunk with size 0 or 0xFFFFFFFF (streaming writers)
// extends to the end of the stream.
func Read(r io.Reader) ([]int16, error) {
	var header [12]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNotWAV, err)
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, ErrNotWAV
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
			if channels != Channels {
				return nil, fmt.Errorf("wav: %d channels, want %d", channels, Channels)
			}
			if rate != SampleRate {
				return nil, fmt.Errorf("wav: sample rate %d Hz, want %d", rate, SampleRate)
			}
			if bits != Bits {
				return nil, fmt.Errorf("wav: %d bits per sample, want %d", bits, Bits)
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
			return DecodePCM16(raw), nil
		default:
			if _, err := io.CopyN(io.Discard, r, int64(size+size%2)); err != nil {
				return nil, fmt.Errorf("wav: chunk %q: %w", id, err)
			}
		}
	}
}

// DecodePCM16 converts little-endian 16-bit samples; a trailing odd byte is dropped.
func DecodePCM16(raw []byte) []int16 {
	pcm := make([]int16, len(raw)/2)
	for i := range pcm {
		pcm[i] = int16(binary.LittleEndian.Uint16(raw[2*i:]))
	}
	return pcm
}

// Write encodes samples as a 16 kHz mono 16-bit PCM WAV.
func Write(w io.Writer, pcm []int16) error {
	data := make([]byte, 2*len(pcm))
	for i, s := range pcm {
		binary.LittleEndian.PutUint16(data[2*i:], uint16(s))
	}
	var hdr [44]byte
	copy(hdr[0:4], "RIFF")
	binary.LittleEndian.PutUint32(hdr[4:8], uint32(36+len(data)))
	copy(hdr[8:12], "WAVE")
	copy(hdr[12:16], "fmt ")
	binary.LittleEndian.PutUint32(hdr[16:20], 16)
	binary.LittleEndian.PutUint16(hdr[20:22], 1)
	binary.LittleEndian.PutUint16(hdr[22:24], Channels)
	binary.LittleEndian.PutUint32(hdr[24:28], SampleRate)
	binary.LittleEndian.PutUint32(hdr[28:32], SampleRate*Channels*Bits/8)
	binary.LittleEndian.PutUint16(hdr[32:34], Channels*Bits/8)
	binary.LittleEndian.PutUint16(hdr[34:36], Bits)
	copy(hdr[36:40], "data")
	binary.LittleEndian.PutUint32(hdr[40:44], uint32(len(data)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(data)
	return err
}
