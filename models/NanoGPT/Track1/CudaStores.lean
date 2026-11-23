import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Models.NanoGPT.Track1.CudaStores

abbrev Memory := Nat → UInt8

structure Store where
  address : Nat
  word : UInt8

def commit (store : Store) (mem : Memory) : Memory := fun address =>
  if address = store.address then store.word else mem address

def commitAll : List Store → Memory → Memory
  | [], mem => mem
  | store :: rest, mem => commitAll rest (commit store mem)

theorem preserve_unwritten (stores : List Store) (mem : Memory) (addr : Nat)
    (unwritten : ∀ w ∈ stores, w.address ≠ addr) :
    commitAll stores mem addr = mem addr := by
  induction stores generalizing mem with
  | nil => rfl
  | cons first rest ih =>
    have away := unwritten first (List.mem_cons_self ..)
    have other (w : Store) (inRest : w ∈ rest) : w.address ≠ addr :=
      unwritten w (List.mem_cons_of_mem first inRest)
    simp [commitAll, ih _ other, commit, Ne.symm away]

theorem unique_store_result (stores : List Store) (mem : Memory) (item : Store)
    (unique : (stores.map Store.address).Nodup) (written : item ∈ stores) :
    commitAll stores mem item.address = item.word := by
  induction stores generalizing mem with
  | nil => simp at written
  | cons first rest ih =>
    have ⟨notInRest, uniqueRest⟩ := List.nodup_cons.mp unique
    rcases List.mem_cons.mp written with eq | later
    · subst first
      have outside (w : Store) (inRest : w ∈ rest) : w.address ≠ item.address := by
        intro same
        have yes : item.address ∈ rest.map Store.address := List.mem_map.mpr ⟨w, inRest, same⟩
        exact notInRest yes
      simpa [commitAll, commit] using preserve_unwritten rest (commit item mem) item.address outside
    · exact ih (commit first mem) uniqueRest later

theorem unique_stores_commute (stores other : List Store) (mem : Memory)
    (unique : (stores.map Store.address).Nodup) (same : stores.Perm other) :
    commitAll stores mem = commitAll other mem := by
  funext addr
  by_cases written : ∃ w ∈ stores, w.address = addr
  · obtain ⟨w, member, atAddr⟩ := written
    have member' := same.mem_iff.mp member
    have unique' := (same.map Store.address).nodup_iff.mp unique
    rw [← atAddr, unique_store_result _ _ _ unique member,
      unique_store_result _ _ _ unique' member']
  · have away (w : Store) (member : w ∈ stores) : w.address ≠ addr :=
      fun h => written ⟨w, member, h⟩
    have away' (w : Store) (member : w ∈ other) : w.address ≠ addr :=
      away w (same.mem_iff.mpr member)
    rw [preserve_unwritten _ _ _ away, preserve_unwritten _ _ _ away']

structure Block where
  read : Nat → Prop
  stores : Memory → List Store
  readsOnly : ∀ mem next : Memory, (∀ a, read a → mem a = next a) → stores mem = stores next

def execute : List Block → Memory → Memory
  | [], mem => mem
  | block :: rest, mem => execute rest (commitAll (block.stores mem) mem)

theorem commit_append (first next : List Store) (mem : Memory) :
    commitAll (first ++ next) mem = commitAll next (commitAll first mem) := by
  induction first generalizing mem with
  | nil => rfl
  | cons item rest ih => exact ih _

theorem independent_block_reads (blocks : List Block) (initial mem : Memory)
    (readable : ∀ block ∈ blocks, ∀ addr, block.read addr → mem addr = initial addr)
    (independent : ∀ writer ∈ blocks, ∀ w ∈ writer.stores initial,
      ∀ reader ∈ blocks, ¬ reader.read w.address) :
    execute blocks mem = commitAll (blocks.flatMap (·.stores initial)) mem := by
  induction blocks generalizing mem with
  | nil => rfl
  | cons block rest ih =>
    have this : block.stores mem = block.stores initial :=
      block.readsOnly _ _ (readable block (List.mem_cons_self ..))
    rw [execute, this]
    have nextReadable (reader : Block) (member : reader ∈ rest) (addr : Nat) (reads : reader.read addr) :
        commitAll (block.stores initial) mem addr = initial addr := by
      have member' := List.mem_cons_of_mem block member
      rw [preserve_unwritten]
      · exact readable reader member' addr reads
      · intro w inBlock atAddr
        exact independent block (List.mem_cons_self ..) w inBlock reader member' (atAddr ▸ reads)
    have restIndependent (writer : Block) (writerLive : writer ∈ rest) (w : Store)
        (inBlock : w ∈ writer.stores initial) (reader : Block) (readerLive : reader ∈ rest) :
        ¬ reader.read w.address :=
      independent writer (List.mem_cons_of_mem _ writerLive) w inBlock reader
        (List.mem_cons_of_mem _ readerLive)
    rw [ih _ nextReadable restIndependent, List.flatMap_cons, commit_append]

theorem launch_order_independent (blocks scheduled : List Block) (initial : Memory)
    (schedule : blocks.Perm scheduled)
    (independent : ∀ writer ∈ blocks, ∀ w ∈ writer.stores initial,
      ∀ reader ∈ blocks, ¬ reader.read w.address)
    (unique : ((blocks.flatMap (·.stores initial)).map Store.address).Nodup) :
    execute blocks initial = execute scheduled initial := by
  have scheduledIndependent : ∀ writer ∈ scheduled, ∀ w ∈ writer.stores initial,
      ∀ reader ∈ scheduled, ¬ reader.read w.address := by
    intro writer hasWriter w produced reader hasReader
    exact independent writer (schedule.mem_iff.mpr hasWriter) w produced reader
      (schedule.mem_iff.mpr hasReader)
  rw [independent_block_reads blocks initial initial (fun _ _ _ _ => rfl) independent,
    independent_block_reads scheduled initial initial (fun _ _ _ _ => rfl) scheduledIndependent]
  exact unique_stores_commute _ _ initial unique (schedule.flatMap_right (·.stores initial))

#axiomsClean launch_order_independent
#axiomsClean unique_stores_commute
#axiomsClean independent_block_reads

end VeriTile.Bench.Models.NanoGPT.Track1.CudaStores
