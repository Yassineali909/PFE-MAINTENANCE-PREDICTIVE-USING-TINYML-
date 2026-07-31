import os
import re
from pathlib import Path

def patch_compatibility(source, target, env):
    # Chemin vers compatibility.h dans la bibliothèque téléchargée
    compat_path = os.path.join(
        env.subst("$PROJECT_LIBDEPS_DIR"),
        env.subst("$PIOENV"),
        "Arduino_TensorFlowLite",
        "src", "tensorflow", "lite", "micro", "compatibility.h"
    )

    if not os.path.exists(compat_path):
        print(f"[patch_tflm] Fichier non trouvé: {compat_path}")
        return

    with open(compat_path, "r") as f:
        content = f.read()

    # Vérifie si déjà patché (notre marqueur)
    if "PATCHED_BY_PFE" in content:
        print("[patch_tflm] Déjà patché — rien à faire.")
        return

    # Patch 1 : transformer le delete privé en public
    # Recherche le pattern "private:" suivi de "void operator delete(void* p) {}"
    patched = re.sub(
        r'(private:\s*)void operator delete\(void\* p\) \{\}',
        r'public:\n  void operator delete(void* p) {} // PATCHED_BY_PFE',
        content,
        flags=re.DOTALL
    )

    # Si le pattern n'a pas fonctionné, on tente une approche plus simple
    if patched == content:
        patched = content.replace(
            "void operator delete(void* p) {}",
            "void operator delete(void* p) {} // PATCHED_BY_PFE"
        )
        # On rend la ligne non privée en supprimant le "private:" qui précède
        patched = patched.replace("private:\n  // PATCHED_BY_PFE", "public:\n  ")

    if patched != content:
        with open(compat_path, "w") as f:
            f.write(patched)
        print(f"[patch_tflm] ✅ compatibility.h patché avec succès.")
    else:
        print("[patch_tflm] ⚠️ Aucun changement effectué — vérifiez le contenu du fichier.")

# Enregistre le script pour qu'il s'exécute avant la compilation
def register():
    from SCons.Script import Import
    Import("env")
    env.AddPreAction("buildprog", patch_compatibility)

# Appel automatique lorsque PlatformIO charge le script
register()
