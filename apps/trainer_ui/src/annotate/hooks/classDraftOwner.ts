// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
/**
 * A class draft belongs to the project it was loaded from, and may only be
 * written back to that one.
 *
 * Five places write the draft to the server and four of them used to pair "the
 * project open right now" with "whatever is in the draft ref". Those are the
 * same thing only once a switch has finished loading; until then the ref still
 * holds the previous project's list, and every one of those four writes is a
 * fire-and-forget flush that happens exactly around a switch.
 *
 * The gap is not harmless, because the server accepts a superset of the ids a
 * project already has. Another project's list therefore does not fail
 * validation: it renames the class the masks were painted with and adds the
 * foreign ones next to it. Correcting it afterwards is refused, since dropping
 * those ids is a removal. A project can then train a full run on a class list
 * belonging to an unrelated dataset, with scores comparable to nothing.
 *
 * So the draft carries the id of the project it came from, and a write that
 * does not match is dropped instead of sent.
 */
import type React from "react";
import { saveClasses } from "../../api";
import type { ClassItem } from "../../store";

export type ClassDraftOwnerRef = React.MutableRefObject<string | null>;

type SavedClasses = { classes?: ClassItem[] } | undefined;

/** Whether the draft currently held was loaded from *projectId*. */
export function classDraftBelongsTo(
  ownerRef: ClassDraftOwnerRef,
  projectId: string | null,
): boolean {
  return !!projectId && ownerRef.current === projectId;
}

export function classDraftPayload(draft: ClassItem[], nextClassId: number) {
  return {
    version: 1,
    ignore_index: 255,
    classes: draft,
    next_class_id: nextClassId,
  };
}

/**
 * Save *draft* to *projectId*, or nothing at all when it belongs elsewhere.
 *
 * Returns null in that case rather than throwing: every caller is a flush on
 * the way out of somewhere, and there is nobody left to handle an error.
 */
export function saveClassDraft(
  ownerRef: ClassDraftOwnerRef,
  projectId: string | null,
  draft: ClassItem[],
  nextClassId: number,
): Promise<SavedClasses> | null {
  if (!projectId || draft.length === 0) return null;
  if (!classDraftBelongsTo(ownerRef, projectId)) {
    console.warn(
      `classes: draft belongs to ${ownerRef.current ?? "no project"}, refusing to write it to ${projectId}`,
    );
    return null;
  }
  return saveClasses(projectId, classDraftPayload(draft, nextClassId)) as Promise<SavedClasses>;
}
