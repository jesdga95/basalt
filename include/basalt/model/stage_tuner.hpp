// include/basalt/model/stage_tuner.hpp - where the main card hands a long prompt to its stage card.
//
// The shares give the first cut a run reads with; a prompt's chunks cross to the stage card at a layer, and the stage
// card is the long pole of the pipeline when its layers cost more than the main card's. The tuner keeps the cut where
// the pipeline of the last long read says the read is shortest, never giving the stage card more layers than the shares
// did. Pure arithmetic over what one read measured: no GPU calls, nothing priced ahead of a read.
#pragma once

namespace basalt::model {

class StageTuner {
public:
    StageTuner() = default;
    /// The layers [0, end) are read by the main card and one stage card; `widest` is the stage's first layer when the shares
    /// cut it (the most layers it may read). A tuner that is not `active` keeps that cut.
    StageTuner(int end, int widest, bool active) : end_(end), widest_(widest), cut_(widest), active_(active) {}

    /// The stage's first layer for the next read.
    int cut() const { return cut_; }

    /// One read's result: the stage read layers [cut, end) of `chunks` chunks in `stage_ms` of work, the whole read took
    /// `total_ms`. Moves the cut to the layer that makes that pipeline shortest; returns whether it moved. A read of
    /// fewer than 4 chunks is mostly pipeline fill and tells little: it changes nothing.
    bool observe(int cut, int chunks, double total_ms, double stage_ms);

    /// A read of `chunks` chunks when a layer costs the main card `a` ms and the stage card `b` ms per chunk, the stage
    /// reading `n1` layers: the stage's chunks one after another behind the main card's, then the main card's last chunk
    /// through the stage's layers.
    double read_ms(int n1, double a, double b, int chunks) const;

private:
    int end_ = 0, widest_ = 0, cut_ = 0;
    bool active_ = false;
};

}  // namespace basalt::model
