import models.NanoGPT.Track1.SparseLanding

/-! Cut the sorted request lists by owner and interpret the two row
exchanges. A rank sends IDs in destination order. Owners answer in
source order, using the transposed counts for the return trip. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.RowExchange
open NgramSlots (Want IsInsertion nativeNat native_insertion)
open SparsePackets (Cut Channel signed_local_offset)

def narrowKey (x : Nat) : Nat := (Int32.ofNat x).toInt.toNat

theorem narrow_key_exact (x : Nat) (fits : x < 2 ^ 31) : narrowKey x = x := by
  simp [narrowKey, Int32.toInt_ofNat_of_lt fits]

def narrowWant (want : Want) : Want where
  size := want.size
  read i := narrowKey (want.read i)
  ordered i j order live := by
    rw [narrow_key_exact _ (Nat.lt_trans (want.inVocab i (by omega)) (by decide)),
      narrow_key_exact _ (Nat.lt_trans (want.inVocab j live) (by decide))]
    exact want.ordered i j order live
  inVocab i live := by
    rw [narrow_key_exact _ (Nat.lt_trans (want.inVocab i live) (by decide))]
    exact want.inVocab i live

/-- Lower-bound searches see the same signed keys. This covers the
int32 path without requiring an int64 copy of the wanted row array. -/
theorem compact_cuts_exact (want : Want) (boundary : Nat) (fits : boundary < 2 ^ 31) :
    nativeNat (narrowWant want) (narrowKey boundary) = nativeNat want boundary := by
  rw [narrow_key_exact _ fits]
  have narrowed := native_insertion (narrowWant want) boundary
  rcases narrowed with ⟨lo, hi, lower, upper⟩
  apply NgramSlots.insertion_unique want.read boundary 0 want.size _ _ _ (native_insertion want boundary)
  refine ⟨lo, hi, ?_, ?_⟩
  · intro i origin before
    have live : i < want.size := Nat.lt_of_lt_of_le before hi
    simpa only [narrowWant, narrow_key_exact _ (Nat.lt_trans (want.inVocab i live) (by decide))] using
      lower i origin before
  · intro i after live
    simpa only [narrowWant, narrow_key_exact _ (Nat.lt_trans (want.inVocab i live) (by decide))] using
      upper i after live

def shardRows : Nat := 84602880 / 8
def boundary (rank : Nat) : Nat := rank * shardRows
def cutAt (want : Want) (rank : Nat) : Nat := nativeNat want (boundary rank)

theorem lower_bounds_monotone (want : Want) : Monotone (nativeNat want) := by
  intro a b bounds
  by_contra cast
  have left := native_insertion want a
  have right := native_insertion want b
  have x := left.2.2.1 (nativeNat want b) (Nat.zero_le _) (Nat.lt_of_not_ge cast)
  have y := right.2.2.2 (nativeNat want b) (Nat.le_refl _) (Nat.lt_of_lt_of_le (Nat.lt_of_not_ge cast) left.2.1)
  omega

theorem cuts_monotone (want : Want) : Monotone (cutAt want) := fun _ _ ord =>
  lower_bounds_monotone want (Nat.mul_le_mul_right shardRows ord)

/-- The same host lower-bound cuts rebuilt for the value table's own
shard interval and the original full gradient allocation. -/
def valueSource (want : Want) (sourceRows first rows : Nat)
    (fits : sourceRows < 2 ^ 31) (live : ∀ i, i < want.size → want.read i < sourceRows)
    (unique : Function.Injective (fun i : Fin want.size => want.read i.val))
    (grad : SparsePackets.FP16Rows) (asked : Array Int32) : ValueAdam.SourceSlice :=
  let l := nativeNat want first
  let u := nativeNat want (first + rows)
  let cut : Cut := ⟨want.size, l, u, lower_bounds_monotone want (Nat.le_add_right ..),
    (native_insertion want (first + rows)).2.1⟩
  { input := ⟨cut, sourceRows, fun i => ⟨want.read i.val, live i.val i.isLt⟩,
      fun _ _ same => unique (congrArg Fin.val same), grad⟩
    first := first
    shardRows := rows
    asked := asked
    selected := by
      intro slot
      let idx := SparsePackets.ownSlot cut slot
      change first ≤ want.read idx.val ∧ want.read idx.val < first + rows
      have within := slot.isLt
      change slot.val < u - l at within
      have low : l ≤ idx.val := by simp [idx, SparsePackets.ownSlot, cut]
      have high : idx.val < u := by
        change l + slot.val < u
        omega
      exact ⟨(native_insertion want first).2.2.2 _ low idx.isLt,
        (native_insertion want (first + rows)).2.2.1 _ (Nat.zero_le _) high⟩
    signed := fits }

theorem endpoint_cuts (want : Want) : cutAt want 0 = 0 ∧ cutAt want 8 = want.size := by
  constructor
  · apply NgramSlots.insertion_unique _ _ _ _ _ _ (native_insertion want (boundary 0))
    exact ⟨by omega, by omega, fun _ _ h => (Nat.not_lt_zero _ h).elim,
      fun _ _ _ => Nat.zero_le _⟩
  · apply NgramSlots.insertion_unique _ _ _ _ _ _ (native_insertion want (boundary 8))
    exact ⟨by omega, by omega, fun i _ h => want.inVocab i h,
      fun _ first last => False.elim (Nat.not_lt_of_ge first last)⟩

def localCut (want : Want) (rank : Nat) : Cut :=
  ⟨want.size, cutAt want rank, cutAt want (rank + 1),
    cuts_monotone want (Nat.le_succ _), (native_insertion want _).2.1⟩

theorem slot_owner (want : Want) (slot rank : Nat) (live : slot < want.size) :
    ((localCut want rank).lo ≤ slot ∧ slot < (localCut want rank).hi) ↔
      boundary rank ≤ want.read slot ∧ want.read slot < boundary (rank + 1) := by
  have lo := native_insertion want (boundary rank)
  have hi := native_insertion want (boundary (rank + 1))
  change cutAt want rank ≤ slot ∧ slot < cutAt want (rank + 1) ↔ _
  constructor
  · intro picked
    exact ⟨lo.2.2.2 slot picked.1 live, hi.2.2.1 slot (Nat.zero_le _) picked.2⟩
  · intro picked
    constructor
    · by_contra h
      exact (Nat.not_lt_of_ge picked.1) (lo.2.2.1 slot (Nat.zero_le _) (Nat.lt_of_not_ge h))
    · by_contra h
      exact (Nat.not_lt_of_ge (hi.2.2.2 slot (Nat.le_of_not_gt h) live)) picked.2

def packetSlots (want : Want) (source owner : Nat) : List Nat :=
  let cut := localCut want owner
  if source = owner then [] else List.range' cut.lo cut.own

def sendCounts (want : Nat → Want) (source owner : Nat) : Nat :=
  if source = owner then 0 else (localCut (want source) owner).own

def idPacket (want : Nat → Want) (source owner : Nat) : List Nat :=
  (packetSlots (want source) source owner).map (want source).read

theorem packet_length (want : Nat → Want) (source owner : Nat) :
    (idPacket want source owner).length = sendCounts want source owner := by
  simp only [idPacket, packetSlots, sendCounts, List.length_map]
  split <;> simp

theorem id_in_owner (want : Nat → Want) (source owner row : Nat) (member : row ∈ idPacket want source owner) :
    boundary owner ≤ row ∧ row < boundary (owner + 1) := by
  obtain ⟨slot, inside, rfl⟩ := List.mem_map.mp member
  unfold packetSlots at inside
  split at inside
  · simp at inside
  · obtain ⟨i, within, offset⟩ := List.mem_range'.mp inside
    let cut := localCut (want source) owner
    have pick : cut.lo ≤ slot ∧ slot < cut.hi := by
      change i < cut.hi - cut.lo at within
      change slot = cut.lo + 1 * i at offset
      omega
    exact (slot_owner _ _ _ (Nat.lt_of_lt_of_le pick.2 cut.within)).mp pick

private theorem joinRanges {a b c : Nat} (left : a ≤ b) (right : b ≤ c) :
    List.range' a (b - a) ++ List.range' b (c - b) = List.range' a (c - a) := by
  have next : a + (b - a) = b := by omega
  have total : b - a + (c - b) = c - a := by omega
  simpa only [next, total] using (List.range'_append_1 (s := a) (m := b - a) (n := c - b))

private theorem packet_slots_join (want : Want) (source n : Nat) :
    (List.range n).flatMap (packetSlots want source) =
      List.range (cutAt want (min source n)) ++
        List.range' (cutAt want (source + 1)) (cutAt want n - cutAt want (source + 1)) := by
  induction n with
  | zero => simp [(endpoint_cuts want).1]
  | succ n ih =>
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton, ih]
    have step := cuts_monotone want (Nat.le_succ n)
    by_cases before : n < source
    · have h₀ : min source n = n := Nat.min_eq_right (by omega)
      have h₁ : min source (n + 1) = n + 1 := Nat.min_eq_right before
      have s₀ : cutAt want n - cutAt want (source + 1) = 0 :=
        Nat.sub_eq_zero_of_le (cuts_monotone want (by omega))
      have s₁ : cutAt want (n + 1) - cutAt want (source + 1) = 0 :=
        Nat.sub_eq_zero_of_le (cuts_monotone want (by omega))
      simp only [h₀, h₁, s₀, s₁, List.range'_zero, List.append_nil, packetSlots,
        if_neg (show source ≠ n by omega), localCut, Cut.own, List.range_eq_range']
      simpa only [Nat.sub_zero] using joinRanges (Nat.zero_le (cutAt want n)) step
    · by_cases eq : source = n
      · subst source
        have empty : cutAt want n - cutAt want (n + 1) = 0 := Nat.sub_eq_zero_of_le step
        simp [packetSlots, empty]
      · have past : source + 1 ≤ n := by omega
        simp only [Nat.min_eq_left (show source ≤ n by omega), Nat.min_eq_left (show source ≤ n + 1 by omega),
          packetSlots, if_neg eq, localCut, Cut.own, List.append_assoc]
        rw [joinRanges (cuts_monotone want past) step]

theorem peer_order (want : Want) (rank : Fin 8) :
    (List.range 8).flatMap (packetSlots want rank) =
      List.range (localCut want rank).lo ++ List.range' (localCut want rank).hi (want.size - (localCut want rank).hi) := by
  rw [packet_slots_join, Nat.min_eq_left (Nat.le_of_lt rank.isLt), (endpoint_cuts want).2]
  rfl

abbrev Banks := Nat → Nat → Channel → UInt16

def globalWord (banks : Banks) (row : Nat) (col : Channel) : UInt16 :=
  banks (row / shardRows) (row % shardRows) col

def recvCounts (wanted : Nat → Want) (owner source : Nat) : Nat :=
  sendCounts wanted source owner

def asked (wanted : Nat → Want) (owner : Nat) : List Nat :=
  (List.range 8).flatMap fun source => idPacket wanted source owner

def served (wanted : Nat → Want) (banks : Banks) (owner : Nat) (col : Channel) : List UInt16 :=
  (asked wanted owner).map fun row => banks owner (row - boundary owner) col

def returnSegment (wanted : Nat → Want) (banks : Banks) (owner source : Nat) (col : Channel) : List UInt16 :=
  let start := ((List.range source).map (recvCounts wanted owner)).sum
  ((served wanted banks owner col).drop start).take (recvCounts wanted owner source)

/-- The reply uses the saved receive counts for its send splits. The
matching receive splits concatenate the owner segments in rank order. -/
def returnWords (wanted : Nat → Want) (banks : Banks) (source : Nat) (col : Channel) : List UInt16 :=
  (List.range 8).flatMap fun owner => returnSegment wanted banks owner source col

theorem return_segment_exact (wanted : Nat → Want) (banks : Banks) (owner : Nat) (source : Fin 8)
    (col : Channel) : returnSegment wanted banks owner source col =
      (idPacket wanted source owner).map (fun row => banks owner (row - boundary owner) col) := by
  let prefixIds := (List.range source.val).flatMap fun peer => idPacket wanted peer owner
  have splitRanks : List.range 8 = (List.range source.val ++ [source.val]) ++
      (List.range (8 - (source.val + 1))).map (source.val + 1 + ·) := by
    have fits := source.isLt
    have ranks : source.val + 1 + (8 - (source.val + 1)) = 8 := by omega
    simpa only [ranks, List.range_succ (n := source.val)] using
      (List.range_add (n := source.val + 1) (m := 8 - (source.val + 1)))
  have start : ((List.range source.val).map (recvCounts wanted owner)).sum = prefixIds.length := by
    simp only [prefixIds, List.length_flatMap, packet_length]
    rfl
  unfold returnSegment served asked
  rw [start]
  simp only [splitRanks, List.flatMap_append, List.flatMap_singleton, List.map_append]
  rw [List.append_assoc]
  change ((List.map _ prefixIds ++ (List.map _ (idPacket wanted source owner) ++ _)).drop prefixIds.length).take
    (recvCounts wanted owner source) = _
  rw [← List.length_map (f := fun row => banks owner (row - boundary owner) col) (as := prefixIds),
    List.drop_append_length]
  have count : recvCounts wanted owner source = ((idPacket wanted source owner).map
    (fun row => banks owner (row - boundary owner) col)).length := by simp [packet_length, recvCounts]
  rw [count, List.take_append_length]

theorem owner_address (owner row : Nat) (range : boundary owner ≤ row ∧ row < boundary (owner + 1)) :
    row / shardRows = owner ∧ row % shardRows = row - boundary owner := by
  unfold boundary shardRows at *
  omega

theorem returned_peer_words (wanted : Nat → Want) (banks : Banks) (source : Fin 8) (col : Channel) :
    returnWords wanted banks source col =
      (List.range (localCut (wanted source) source).lo ++
        List.range' (localCut (wanted source) source).hi ((wanted source).size - (localCut (wanted source) source).hi)).map
          (fun slot => globalWord banks ((wanted source).read slot) col) := by
  unfold returnWords
  rw [← peer_order (wanted source) source, List.map_flatMap]
  apply List.flatMap_congr
  intro owner _
  rw [return_segment_exact wanted banks owner source col]
  simp only [idPacket, List.map_map, Function.comp_def]
  apply List.map_congr_left
  intro slot member
  have sent : (wanted source).read slot ∈ idPacket wanted source owner := List.mem_map_of_mem member
  obtain ⟨found, offset⟩ := owner_address owner _ (id_in_owner wanted source owner _ sent)
  simp [globalWord, found, offset]

def landingInput (wanted : Nat → Want) (banks : Banks) (rank : Fin 8)
    (previous : SparseLanding.Rows) : SparseLanding.LandInput :=
  let want := wanted rank
  ⟨localCut want rank,
    ⟨want.size, fun i => Int32.ofNat (want.read i), Int32.ofNat (boundary rank), shardRows, banks rank⟩,
    fun i col => ((returnWords wanted banks rank col)[i]?).getD 0,
    previous⟩

private theorem wireWord (wanted : Nat → Want) (banks : Banks) (rank : Fin 8)
    (slot : Nat) (col : Channel) (live : slot < (wanted rank).size)
    (remote : slot < (localCut (wanted rank) rank).lo ∨ (localCut (wanted rank) rank).hi ≤ slot) :
    let cut := localCut (wanted rank) rank
    let wire := slot - (if cut.hi ≤ slot then cut.own else 0)
    (returnWords wanted banks rank col)[wire]? = some (globalWord banks ((wanted rank).read slot) col) := by
  intro cut wire
  rw [returned_peer_words wanted banks rank col, List.getElem?_map]
  rcases remote with before | suffix
  · change slot < cut.lo at before
    have atWire : wire = slot := by
      simp only [wire, if_neg (show ¬cut.hi ≤ slot by have := cut.ordered; omega), Nat.sub_zero]
    rw [atWire, List.getElem?_append_left (by simpa only [List.length_range] using before), List.getElem?_range before]
    rfl
  · change cut.hi ≤ slot at suffix
    have next : cut.lo ≤ wire := by
      have := cut.ordered
      simp only [wire, if_pos suffix, Cut.own]
      omega
    rw [List.getElem?_append_right (by simpa only [List.length_range] using next), List.length_range]
    have offset : wire - cut.lo = slot - cut.hi := by
      have := cut.ordered
      simp only [wire, if_pos suffix, Cut.own]
      omega
    rw [offset, List.getElem?_range' (by omega : slot - cut.hi < (wanted rank).size - cut.hi),
      Nat.one_mul, Nat.add_sub_of_le suffix]
    rfl

private theorem readOwned (banks : Banks) (rank row : Nat) (col : Channel) (shard : SparseLanding.ServeInput)
    (size : shard.shardRows = shardRows) (bank : shard.shard = banks rank) (signed : row < 2 ^ 31)
    (range : boundary rank ≤ row ∧ row < boundary (rank + 1)) :
    SparseLanding.shardRead shard (Int32.ofNat row - Int32.ofNat (boundary rank)) col =
      some (globalWord banks row col) := by
  obtain ⟨owner, offset⟩ := owner_address rank row range
  have top : row < boundary rank + shardRows := by
    simpa only [boundary, Nat.add_mul, Nat.one_mul] using range.2
  have remaining : row - boundary rank < shardRows := by omega
  have inside : 0 ≤ ((row - boundary rank : Nat) : Int) ∧
      ((row - boundary rank : Nat) : Int) < (shardRows : Int) :=
    ⟨Int.natCast_nonneg _, Int.ofNat_lt.mpr remaining⟩
  unfold SparseLanding.shardRead
  rw [signed_local_offset row _ signed range.1, size, if_pos inside, Int.toNat_natCast, bank, globalWord,
    owner, offset]

def serveInput (wanted : Nat → Want) (banks : Banks) (owner : Fin 8) : SparseLanding.ServeInput :=
  let ids := asked wanted owner
  ⟨ids.length, fun i => Int32.ofNat (ids[i]?.getD 0), Int32.ofNat (boundary owner), shardRows, banks owner⟩

/-- The shard gather implements the owner's contiguous reply buffer,
including the mask for entries past the received ID packet. -/
theorem served_packet_word (wanted : Nat → Want) (banks : Banks) (owner : Fin 8) (i : Nat) (col : Channel) :
    SparseLanding.directServe (serveInput wanted banks owner) i col = (served wanted banks owner col)[i]? := by
  let ids := asked wanted owner
  unfold SparseLanding.directServe served
  rw [List.getElem?_map]
  by_cases live : i < ids.length
  · rw [if_pos (show i < (serveInput wanted banks owner).count from live), List.getElem?_eq_getElem live]
    let row := ids[i]'live
    have member : row ∈ ids := List.getElem_mem live
    obtain ⟨source, _, requested⟩ := List.mem_flatMap.mp member
    have range := id_in_owner wanted source owner row requested
    have safe : row < 2 ^ 31 := by
      have width : boundary (owner.val + 1) ≤ boundary 8 := Nat.mul_le_mul_right shardRows owner.isLt
      have endSafe : boundary 8 < 2 ^ 31 := by decide
      exact Nat.lt_trans range.2 (Nat.lt_of_le_of_lt width endSafe)
    change SparseLanding.shardRead (serveInput wanted banks owner)
      (Int32.ofNat (((asked wanted owner)[i]?).getD 0) - Int32.ofNat (boundary owner)) col =
      some (banks owner (row - boundary owner) col)
    rw [List.getElem?_eq_getElem live, Option.getD_some]
    have bytes := readOwned banks owner row col (serveInput wanted banks owner) rfl rfl safe range
    obtain ⟨home, offset⟩ := owner_address owner row range
    rw [globalWord, home, offset] at bytes
    exact bytes
  · rw [if_neg (show ¬i < (serveInput wanted banks owner).count from live),
      List.getElem?_eq_none (Nat.le_of_not_gt live)]
    rfl

/-- Every active cache slot contains the BF16 word of its global ID in
the post-update owner shard. The transport only moves payload words. -/
theorem landed_cache_word (wanted : Nat → Want) (banks : Banks) (rank : Fin 8)
    (previous : SparseLanding.Rows) (slot : Nat) (col : Channel) (live : slot < (wanted rank).size) :
    SparseLanding.directLand (landingInput wanted banks rank previous) slot col =
      some (globalWord banks ((wanted rank).read slot) col) := by
  let cut := localCut (wanted rank) rank
  let row := (wanted rank).read slot
  change (if slot < cut.rows then
    if cut.lo ≤ slot ∧ slot < cut.hi then
      SparseLanding.shardRead (landingInput wanted banks rank previous).shard
        (Int32.ofNat row - Int32.ofNat (boundary rank)) col
    else some (((returnWords wanted banks rank col)[slot - (if cut.hi ≤ slot then cut.own else 0)]?).getD 0)
    else some (previous slot col)) = _
  rw [if_pos (show slot < cut.rows from live)]
  split
  · rename_i owned
    exact readOwned banks rank row col (landingInput wanted banks rank previous).shard rfl rfl
      (Nat.lt_trans ((wanted rank).inVocab slot live) (by decide)) ((slot_owner (wanted rank) slot rank live).mp owned)
  · rename_i outside
    have remote : slot < cut.lo ∨ cut.hi ≤ slot := by omega
    rw [wireWord wanted banks rank slot col live remote]
    rfl

theorem wanted_member_slot (want : Want) (row : Nat) (member : NgramSlots.ContainsRow want row) :
    nativeNat want row < want.size ∧ want.read (nativeNat want row) = row := by
  obtain ⟨i, inside, value⟩ := member
  have valid := native_insertion want row
  have atBefore : nativeNat want row ≤ i := by
    by_contra later
    exact (Nat.not_lt_of_ge (Nat.le_of_eq value.symm))
      (valid.2.2.1 i (Nat.zero_le _) (Nat.lt_of_not_ge later))
  have live := Nat.lt_of_le_of_lt atBefore inside
  have beforeValue := want.ordered _ _ atBefore inside
  rw [value] at beforeValue
  exact ⟨live, Nat.le_antisymm beforeValue (valid.2.2.2 _ (Nat.le_refl _) live)⟩

#axiomsClean compact_cuts_exact
#axiomsClean returned_peer_words
#axiomsClean served_packet_word
#axiomsClean landed_cache_word
#axiomsClean wanted_member_slot
end VeriTile.Bench.Models.NanoGPT.Track1.RowExchange
