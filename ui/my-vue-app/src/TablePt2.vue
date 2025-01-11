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

const data_2 = [
  {
    name: "Jan",
    total: Math.floor(Math.random() * 20) + 500,
    predicted: Math.floor(Math.random() * 20) + 500,
  },
  {
    name: "Feb",
    total: Math.floor(Math.random() * 20) + 500,
    predicted: Math.floor(Math.random() * 20) + 500,
  },
  {
    name: "Mar",
    total: Math.floor(Math.random() * 20) + 500,
    predicted: Math.floor(Math.random() * 20) + 500,
  },
  {
    name: "Apr",
    total: Math.floor(Math.random() * 20) + 500,
    predicted: Math.floor(Math.random() * 20) + 500,
  },
  {
    name: "May",
    total: Math.floor(Math.random() * 20) + 500,
    predicted: Math.floor(Math.random() * 20) + 500,
  },
  {
    name: "Jun",
    total: Math.floor(Math.random() * 20) + 500,
    predicted: Math.floor(Math.random() * 20) + 500,
  },
  {
    name: "Jul",
    total: Math.floor(Math.random() * 20) + 500,
    predicted: Math.floor(Math.random() * 20) + 500,
  },
];

type DataRecord = {
  x: number;
  y: number;
  y1?: number;
  y2?: number;
};

type DataRecord2 = {
  x: string;
  y: number;
  y1?: number;
};

const x2 = (d: { x2: number }) => d.x2; // Mappatura per X
const y2 = [
  (d: { y1: number }) => d.y1,
  (d: { y2: number }) => d.y2,
  (d: { y3: number }) => d.y3,
]; // Mappatura per Y

const props = defineProps<{ data: DataRecord[]; data_2: DataRecord2[] }>();

/* -------------------- Bar data----------------------- */

const bar_data = [
  { x: 1, y1: 54, y2: 32, y3: 36 },
  { x: 2, y1: 43, y2: 32, y3: 12 },
  { x: 3, y1: 9, y2: 45, y3: 34 },
];

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
      return d.y1; // Restituisce y1 per gli indici 0, 3, 6, ...
    } else if (index % 3 === 1) {
      return d.y2; // Restituisce y2 per gli indici 1, 4, 7, ...
    } else if (index % 3 === 2) {
      return d.y3; // Restituisce y3 per gli indici 2, 5, 8, ...
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

  commentsArray.forEach((commento) => {
    if (commento.sentimento === "positive") {
      sentimentCount[0] += 1;
    } else if (commento.sentimento === "negative") {
      sentimentCount[1] += 1;
    } else if (commento.sentimento === "neutral") {
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
const users = ref<any[]>([]);
const comment = ref<any[]>([]);

const enterCode = async () => {
  const id = codeInput.value.trim(); // Ottieni il valore inserito dall'utente

  if (!id) {
    alert("Inserisci un ID valido!");
    return;
  }

  try {
    const response = await fetch(`http://localhost:8080/get-data/${id}`);

    if (response.ok) {
      const data = await response.json();
      console.log("Dati ricevuti:", data.data);

      users.value = data.data; // Salva gli utenti nello stato
      comment.value = data.data.commenti;

      isLoaded.value = true; // Carica i contenuti
      donut_data = calculateSentiments(comment);
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
                placeholder="Search products ..."
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
            class="aspect-video rounded-xl bg-white/80 border-0 shadow-md flex flex-col items-center gap-2 p-4"
          >
            <VisSingleContainer :data="donut_data">
              <VisTooltip :triggers="donut_triggers" class-name="" />
              <VisDonut
                :radius="80"
                :value="value"
                :cornerRadius="5"
                centralLabel="Label"
              />
            </VisSingleContainer>
            <VisBulletLegend :items="items" />
          </div>
          <div
            class="aspect-video rounded-xl bg-white/80 border-0 shadow-md flex flex-col items-center gap-2 p-4"
          >
            <VisXYContainer :data="bar_data">
              <VisAxis type="x" :x="x" :tickFormat="tickFormat" />
              <VisAxis type="y" />
              <VisTooltip :triggers="bar_triggers" class-name="" />
              <VisGroupedBar :x="x" :y="y" />
            </VisXYContainer>
            <VisBulletLegend :items="items" />
          </div>
          <div
            class="aspect-video rounded-xl bg-white/80 border-0 shadow-md flex flex-col items-center gap-2 p-4"
          >
            <VisXYContainer :data="data">
              <VisAxis type="x" />
              <VisAxis type="y" />
              <VisTooltip :triggers="triggers" class-name="" />
              <VisArea :x="x2" :y="y2" />
            </VisXYContainer>
            <VisBulletLegend :items="items" />
          </div>
        </div>
        <div class="grid auto-rows-min gap-4 md:grid-cols-3">
          <div class="h-[100vh] md:min-h-min col-span-3">
            <Card class="w-full rounded-xl bg-white/80 border-0 shadow-md">
              <CardHeader>
                <CardTitle>Users</CardTitle>
                <CardDescription>
                  Manage your users and view their information.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table class="">
                  <TableHeader>
                    <TableRow>
                      <TableHead>Id</TableHead>
                      <TableHead class="w-full"> Commento </TableHead>
                      <TableHead class="">Sentimento</TableHead>
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
                        {{ comment.comento }}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">
                          {{ comment.sentimento }}
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
