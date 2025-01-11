<script setup lang="ts">
import { ref } from "vue";
import { ArrowDownRight } from "lucide-vue-next";
import { Badge } from "@/app/ui/badge";
import { Button } from "@/components/ui/button";
import { useRouter } from "vue-router";

// Usare il router
const router = useRouter();

// Stato per la visibilità dei bottoni
const secondaryButtonVisible = ref(true);
const codeInputVisible = ref(false);

// Variabile per il testo del primo bottone
const primaryButtonText = ref("Primary Button");

// Stato per gestire l'input del codice
const codeInput = ref("");

// Funzione che viene chiamata al click del primo bottone
const onPrimaryButtonClick = () => {
  if (codeInputVisible.value) {
    // Se l'input è visibile, chiama la funzione active
    enterCode();
  } else {
    // Nascondi il secondo bottone e mostra il campo di input
    secondaryButtonVisible.value = false;
    codeInputVisible.value = true;
    primaryButtonText.value = "Submit"; // Cambia il testo del bottone
  }
};

// Funzione per navigare alla pagina "About"
const goToHome = () => {
  router.push("/home");
};

const enterCode = async () => {
  const id = codeInput.value.trim(); // Ottieni il valore inserito dall'utente

  if (!id) {
    alert("Inserisci un ID valido!");
    return;
  }

  try {
    // Effettua una richiesta GET all'endpoint
    const response = await fetch(`http://localhost:8080/get-data/${id}`);

    if (response.ok) {
      const data = await response.json();
      console.log("Dati ricevuti:", data);

      // Accedi ai dati nel JSON
      if (data && data.data) {
        const user = data.data; // Estrai l'oggetto user

        await router.push({
          path: "/review",
          query: {
            id: user.id,
            username: user.username,
            bio: user.bio,
          },
        });
      } else {
        alert("Dati mancanti o struttura non valida.");
      }
    } else {
      const errorData = await response.json();
      alert(`Errore: ${errorData.detail || "Qualcosa è andato storto"}`);
    }
  } catch (error) {
    console.error("Errore durante la richiesta:", error);
    alert("Errore durante la richiesta. Controlla la connessione e riprova.");
  }
};
</script>

<template>
  <section class="py-32">
    <div class="container">
      <div class="grid items-center gap-8 lg:grid-cols-2">
        <div
          class="flex flex-col items-center text-center lg:items-start lg:text-left"
        >
          <Badge variant="outline">
            New Release
            <ArrowDownRight class="ml-2 size-4" />
          </Badge>
          <h1 class="my-6 text-pretty text-4xl font-bold lg:text-6xl">
            Welcome to Our Website
          </h1>
          <p class="mb-8 max-w-xl text-muted-foreground lg:text-xl">
            Lorem ipsum dolor sit amet consectetur adipisicing elit. Elig
            doloremque mollitia fugiat omnis! Porro facilis quo animi
            consequatur. Explicabo.
          </p>
          <div
            class="flex w-full flex-col justify-center items-center gap-2 sm:flex-row lg:justify-start"
          >
            <div
              class="flex w-full flex-col justify-center items-center gap-2 sm:flex-row lg:justify-start"
            >
              <Button class="w-full sm:w-auto" @click="onPrimaryButtonClick">
                {{ primaryButtonText }}
              </Button>

              <Button
                v-if="secondaryButtonVisible"
                variant="outline"
                class="w-full sm:w-auto"
                @click="goToHome"
              >
                Secondary Button
                <ArrowDownRight class="ml-2 size-4" />
              </Button>

              <input
                v-if="codeInputVisible"
                type="text"
                v-model="codeInput"
                class="w-full sm:w-auto p-2 text-sm border border-gray-300 rounded-md focus:outline-none slide-in"
                placeholder="Enter your code"
              />
            </div>
          </div>
        </div>
        <img
          src="https://shadcnblocks.com/images/block/placeholder-1.svg"
          alt="placeholder hero"
          class="max-h-96 w-full rounded-md object-cover"
        />
      </div>
    </div>
  </section>
</template>

<style scoped>
/* Animazione slide-in da sinistra a destra a partire dalla fine del primo pulsante */
@keyframes slideIn {
  0% {
    transform: translateX(150%); /* Inizia fuori dalla vista, a destra */
  }
  100% {
    transform: translateX(0); /* Posizione finale visibile */
  }
}

.slide-in {
  animation: slideIn 0.5s ease-out;
}
</style>
