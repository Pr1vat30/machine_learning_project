<script setup lang="ts">
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/app/ui/badge";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/app/ui/card";

import {
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/ui/table";

import {
  VisXYContainer,
  VisGroupedBar,
  VisAxis,
  VisArea,
  VisSingleContainer,
  VisDonut,
  VisTooltip,
  VisBulletLegend,
} from "@unovis/vue";

import { Search, ChevronRight } from "lucide-vue-next";
import { onMounted, ref } from "vue";
import { Donut, GroupedBar } from "@unovis/ts";

const Spacing = { top: 16, bottom: 16, left: 16, right: 16 };

/* -------------------- Bar data----------------------- */

function getCommentsThisWeek(jsonData) {
  // Array dei giorni della settimana
  const daysOfWeek = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
  ];

  // Inizializzazione dei contatori per ogni giorno della settimana
  const sentimentCounts = [
    { x: 1, y1: 0, y2: 0, y3: 0 }, // Monday
    { x: 2, y1: 0, y2: 0, y3: 0 }, // Tuesday
    { x: 3, y1: 0, y2: 0, y3: 0 }, // Wednesday
    { x: 4, y1: 0, y2: 0, y3: 0 }, // Thursday
    { x: 5, y1: 0, y2: 0, y3: 0 }, // Friday
    { x: 6, y1: 0, y2: 0, y3: 0 }, // Saturday
    { x: 7, y1: 0, y2: 0, y3: 0 }, // Sunday
  ];

  // Data corrente
  const today = new Date();
  const sevenDaysAgo = new Date();
  sevenDaysAgo.setDate(today.getDate() - 7);

  jsonData.forEach((activity) => {
    Object.keys(activity.comments).forEach((key) => {
      const comment = activity.comments[key];
      const commentDate = new Date(comment.date);

      // Considera solo i commenti negli ultimi 7 giorni
      if (commentDate >= sevenDaysAgo && commentDate <= today) {
        const dayOfWeek = commentDate.getDay(); // 0 = Sunday, 1 = Monday, ..., 6 = Saturday

        // Mappa il giorno della settimana: Sunday = 0, Monday = 1, ..., Saturday = 6
        // Aggiustiamo l'indice per far partire da 1 (Monday = 1)
        const sentimentIndex = dayOfWeek === 0 ? 6 : dayOfWeek - 1;

        // Conta i sentimenti
        if (comment.sentiment === "positive") {
          sentimentCounts[sentimentIndex].y1++;
        } else if (comment.sentiment === "negative") {
          sentimentCounts[sentimentIndex].y2++;
        } else if (comment.sentiment === "neutral") {
          sentimentCounts[sentimentIndex].y3++;
        }
      }
    });
  });

  return sentimentCounts;
}

let bar_data = [{}, {}, {}];

const dayMap = {
  1: "Lunedì",
  2: "Martedì",
  3: "Mercoledì",
  4: "Giovedì",
  5: "Venerdì",
  6: "Sabato",
  7: "Domenica",
};

const tickFormat = (value) => {
  return dayMap[value] || "Giorno non valido";
};

const x = (d) => d.x;
const y = [(d) => d.y1, (d) => d.y2, (d) => d.y3];

const bar_triggers = {
  [GroupedBar.selectors.bar]: (d, index) => {
    // Determina quale barra è stata selezionata e restituisce il valore corretto
    if (index % 3 === 0) {
      if (d.y1 != 0) {
        return d.y1; // Restituisce y1 per gli indici 0, 3, 6, ...
      } else {
        return "0"; // Restituisce "0" esplicito se d.y1 è 0
      }
    } else if (index % 3 === 1) {
      if (d.y2 != 0) {
        return d.y2; // Restituisce y2 per gli indici 1, 4, 7, ...
      } else {
        return "0"; // Restituisce "0" esplicito se d.y2 è 0
      }
    } else if (index % 3 === 2) {
      if (d.y3 != 0) {
        return d.y3; // Restituisce y3 per gli indici 2, 5, 8, ...
      } else {
        return "0"; // Restituisce "0" esplicito se d.y3 è 0
      }
    }
    return null; // Se non corrisponde, non mostra nessun valore
  },
};

const labels = ["positive", "negative", "neutral"];
const items = labels.map((label) => ({ name: label, inactive: true }));

/* -------------------- Donuts data----------------------- */

let donut_data = [0, 0, 0];

const calculateSentiments = (data) => {
  const sentimentCount = [0, 0, 0];
  const commentsArray = Object.values(data.value);

  commentsArray.forEach((comment) => {
    if (comment.sentiment === "positive") {
      sentimentCount[0] += 1;
    } else if (comment.sentiment === "negative") {
      sentimentCount[1] += 1;
    } else if (comment.sentiment === "neutral") {
      sentimentCount[2] += 1;
    }
  });

  return sentimentCount;
};

const value = (d: number) => d;
const donut_triggers = { [Donut.selectors.segment]: (d) => d.data };

/* -------------------- Request functions----------------------- */

const isLoaded = ref(false);

const codeInput = ref([]);
const form = ref<any[]>([]);
const comment = ref<any[]>([]);

const enterCode = async () => {
  const id = codeInput.value.trim(); // Ottieni il valore inserito dall'utente

  if (!id) {
    alert("Inserisci un ID valido!");
    return;
  }

  try {
    const response = await fetch(__API_SERVER__ + "/api/get-data/${id}");

    if (response.ok) {
      const data = await response.json();
      console.log("Dati ricevuti:", data.data);

      form.value = data.data; // Salva gli utenti nello stato
      comment.value = data.data.comments;

      isLoaded.value = true; // Carica i contenuti
      donut_data = calculateSentiments(comment);

      const datat = getCommentsThisWeek([data.data]);
      bar_data = datat;
      console.log(datat);
    } else {
      const errorData = await response.json();
      alert(`Errore: ${errorData.detail || "Qualcosa è andato storto"}`);
    }
  } catch (error) {
    console.error("Errore durante la richiesta:", error);
    alert("Errore durante la richiesta. Controlla la connessione e riprova.");
  }
};

onMounted(() => {});
</script>

<template>
  <div class="flex min-h-screen w-full flex-col bg-muted/40">
    <!-- Barra di ricerca sempre visibile -->
    <div class="flex flex-col sm:gap-4 sm:py-4">
      <main class="grid flex-1 items-start gap-4 p-4 sm:py-0 md:gap-8">
        <div class="flex items-center">
          <form @submit.prevent="enterCode">
            <div class="relative flex flex-row gap-2">
              <Search
                class="absolute left-2.5 top-1.5 h-4 w-4 text-muted-foreground"
              />
              <Input
                type="search"
                v-model="codeInput"
                placeholder="Insert id form... "
                class="ml-0.5 pl-8 sm:w-[200px] md:w-[200px] lg:w-[200px] h-7 border-0 shadow-md"
              />
              <Button
                variant="outline"
                size="icon"
                class="h-7 border-0 shadow-md"
                @click="enterCode"
              >
                <ChevronRight class="w-4 h-4" />
              </Button>
            </div>
          </form>
          <div class="ml-auto flex items-center gap-2"></div>
        </div>
      </main>

      <!-- Contenuti da caricare solo dopo -->
      <div v-if="isLoaded" class="flex flex-1 flex-col gap-4 p-4 pt-0">
        <div class="grid auto-rows-min gap-4 md:grid-cols-3">
          <div
            class="lg:col-span-1 sm:col-span-full rounded-xl bg-white/80 border-0 shadow-md flex flex-col items-center gap-2"
          >
            <VisSingleContainer :data="donut_data" :margin="Spacing">
              <VisTooltip :triggers="donut_triggers" class-name="" />
              <VisDonut
                :radius="80"
                :value="value"
                :cornerRadius="5"
                centralLabel="Label"
              />
            </VisSingleContainer>
            <VisBulletLegend :items="items" class="pb-4 items-center" />
          </div>
          <div
            class="lg:col-span-2 sm:col-span-full rounded-xl bg-white/80 border-0 shadow-md flex flex-col items-center gap-2"
          >
            <VisXYContainer
              :data="bar_data"
              :yDomain="[
                0,
                Math.max(
                  ...bar_data.flatMap((item) => [item.y1, item.y2, item.y3]),
                ) * 2,
              ]"
              :margin="Spacing"
            >
              <VisAxis
                type="x"
                :x="x"
                :tickFormat="tickFormat"
                tickTextFitMode="trim"
                :numTicks="7"
              />
              <VisAxis type="y" />
              <VisTooltip :triggers="bar_triggers" class-name="" />
              <VisGroupedBar :x="x" :y="y" />
            </VisXYContainer>
            <VisBulletLegend :items="items" class="pb-4 items-center" />
          </div>
        </div>
        <div class="grid auto-rows-min gap-4 md:grid-cols-3">
          <div class="h-[100vh] md:min-h-min col-span-3">
            <Card
              class="w-full max-h-screen overflow-y-auto rounded-xl bg-white/80 border-0 shadow-md"
            >
              <CardHeader>
                <CardTitle>Comment</CardTitle>
                <CardDescription>
                  View user comment information.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table class="">
                  <TableHeader>
                    <TableRow>
                      <TableHead>Id</TableHead>
                      <TableHead class="w-full">Comment</TableHead>
                      <TableHead class="">Sentiment</TableHead>
                    </TableRow>
                  </TableHeader>

                  <TableBody>
                    <!-- Itera sui commenti trasformati in un array -->
                    <TableRow
                      v-for="(comment, index) in Object.values(comment)"
                      :key="index"
                    >
                      <TableCell>{{ index + 1 }}</TableCell>
                      <TableCell class="max-w-12 overflow-hidden text-ellipsis">
                        {{ comment.comment }}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">
                          {{ comment.sentiment }}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </CardContent>
              <CardFooter>
                <div class="text-xs text-muted-foreground">
                  Showing <strong>1-10</strong> of
                  <strong>{{ comment.length }}</strong> comment
                </div>
              </CardFooter>
            </Card>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
