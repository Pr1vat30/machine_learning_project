<script setup lang="ts">
import { Avatar, AvatarFallback, AvatarImage } from "@/app/ui/avatar";
import { Badge } from "@/app/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/app/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/ui/table";
import {
  Activity,
  ArrowUpRight,
  MessageSquareHeart,
  Star,
  FileCheck2,
} from "lucide-vue-next";
import { onMounted, ref } from "vue";

function getDominantSentiment(activities) {
  // Inizializza un oggetto per contare i sentimenti globali
  const sentimentCount = {
    positive: 0,
    negative: 0,
    neutral: 0,
  };

  // Conta i sentimenti di tutti i commenti di tutte le attività
  activities.forEach((activity) => {
    if (activity.comments) {
      for (const commentId in activity.comments) {
        const sentiment = activity.comments[commentId].sentiment;
        if (sentimentCount.hasOwnProperty(sentiment)) {
          sentimentCount[sentiment]++;
        }
      }
    }
  });

  const maxCount = Math.max(...Object.values(sentimentCount));

  const dominantSentiments = Object.keys(sentimentCount).filter(
    (sentiment) => sentimentCount[sentiment] === maxCount,
  );

  return dominantSentiments.length === 1
    ? dominantSentiments[0].charAt(0).toUpperCase() +
        dominantSentiments[0].slice(1)
    : dominantSentiments.map(
        (sentiment) => sentiment.charAt(0).toUpperCase() + sentiment.slice(1),
      );
}

function calculateAverageComments(activities) {
  // Calcola il numero totale di commenti
  let totalComments = 0;

  // Calcola il numero di attività
  const totalActivities = activities.length;

  // Somma i commenti per ogni attività
  activities.forEach((activity) => {
    totalComments += Object.keys(activity.comments).length;
  });

  // Calcola la media
  const averageComments = totalComments / totalActivities;

  return averageComments;
}

const forms = ref<any[]>([]);

const enterCode = async () => {
  try {
    // Effettua una richiesta GET all'endpoint
    const response = await fetch(`http://localhost:8080/get-data/`);

    if (response.ok) {
      const data = await response.json();
      console.log("Dati ricevuti:", data.data);

      // Verifica che i dati siano nel formato corretto e salvali
      if (Array.isArray(data.data)) {
        forms.value = data.data;
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

onMounted(() => {
  enterCode();
});
</script>

<template>
  <div class="flex min-h-screen w-full flex-col">
    <main class="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
      <div class="grid gap-4 md:grid-cols-2 md:gap-8 lg:grid-cols-4">
        <Card>
          <CardHeader
            class="flex flex-row items-center justify-between space-y-0 pb-2"
          >
            <CardTitle class="text-sm font-medium"> Total Comment </CardTitle>
            <Star class="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div v-if="forms.length > 0">
              <div class="text-2xl font-bold">
                +{{ forms[0].n_comment || 0 }}
              </div>
              <p class="text-xs text-muted-foreground">
                Retrieved from all the comments to forms
              </p>
            </div>
            <div v-else>
              <div class="text-2xl font-bold">Missing data</div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader
            class="flex flex-row items-center justify-between space-y-0 pb-2"
          >
            <CardTitle class="text-sm font-medium"> Forms created </CardTitle>
            <FileCheck2 class="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div class="text-2xl font-bold">+{{ forms.length || 0 }}</div>
            <p class="text-xs text-muted-foreground">
              Retrieved from all the from created
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader
            class="flex flex-row items-center justify-between space-y-0 pb-2"
          >
            <CardTitle class="text-sm font-medium">
              Overall sentiment
            </CardTitle>
            <MessageSquareHeart class="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div class="text-2xl font-bold">
              {{ getDominantSentiment(forms) || "null" }}
            </div>
            <p class="text-xs text-muted-foreground">
              Retrieved from all the comment received
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader
            class="flex flex-row items-center justify-between space-y-0 pb-2"
          >
            <CardTitle class="text-sm font-medium">
              Average number of comment
            </CardTitle>
            <Activity class="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div class="text-2xl font-bold">
              {{ calculateAverageComments(forms) || 0 }}
            </div>
            <p class="text-xs text-muted-foreground">
              Average across all forms
            </p>
          </CardContent>
        </Card>
      </div>
      <div class="grid gap-4 md:gap-8 lg:grid-cols-2 xl:grid-cols-3">
        <Card class="xl:col-span-2">
          <CardHeader class="flex flex-row items-center">
            <div class="grid gap-2">
              <CardTitle>Comments</CardTitle>
              <CardDescription> Recent comments</CardDescription>
            </div>
            <Button as-child size="sm" class="ml-auto gap-1">
              <a href="#">
                View All
                <ArrowUpRight class="h-4 w-4" />
              </a>
            </Button>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Customer</TableHead>
                  <TableHead class="hidden xl:table-column"> Type </TableHead>
                  <TableHead class="hidden xl:table-column"> Status </TableHead>
                  <TableHead class="hidden xl:table-column"> Date </TableHead>
                  <TableHead class="text-right"> Amount </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow>
                  <TableCell>
                    <div class="font-medium">Liam Johnson</div>
                    <div class="hidden text-sm text-muted-foreground md:inline">
                      liam@example.com
                    </div>
                  </TableCell>
                  <TableCell class="hidden xl:table-column"> Sale </TableCell>
                  <TableCell class="hidden xl:table-column">
                    <Badge class="text-xs" variant="outline"> Approved </Badge>
                  </TableCell>
                  <TableCell
                    class="hidden md:table-cell lg:hidden xl:table-column"
                  >
                    2023-06-23
                  </TableCell>
                  <TableCell class="text-right"> $250.00 </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <div class="font-medium">Olivia Smith</div>
                    <div class="hidden text-sm text-muted-foreground md:inline">
                      olivia@example.com
                    </div>
                  </TableCell>
                  <TableCell class="hidden xl:table-column"> Refund </TableCell>
                  <TableCell class="hidden xl:table-column">
                    <Badge class="text-xs" variant="outline"> Declined </Badge>
                  </TableCell>
                  <TableCell
                    class="hidden md:table-cell lg:hidden xl:table-column"
                  >
                    2023-06-24
                  </TableCell>
                  <TableCell class="text-right"> $150.00 </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <div class="font-medium">Noah Williams</div>
                    <div class="hidden text-sm text-muted-foreground md:inline">
                      noah@example.com
                    </div>
                  </TableCell>
                  <TableCell class="hidden xl:table-column">
                    Subscription
                  </TableCell>
                  <TableCell class="hidden xl:table-column">
                    <Badge class="text-xs" variant="outline"> Approved </Badge>
                  </TableCell>
                  <TableCell
                    class="hidden md:table-cell lg:hidden xl:table-column"
                  >
                    2023-06-25
                  </TableCell>
                  <TableCell class="text-right"> $350.00 </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <div class="font-medium">Emma Brown</div>
                    <div class="hidden text-sm text-muted-foreground md:inline">
                      emma@example.com
                    </div>
                  </TableCell>
                  <TableCell class="hidden xl:table-column"> Sale </TableCell>
                  <TableCell class="hidden xl:table-column">
                    <Badge class="text-xs" variant="outline"> Approved </Badge>
                  </TableCell>
                  <TableCell
                    class="hidden md:table-cell lg:hidden xl:table-column"
                  >
                    2023-06-26
                  </TableCell>
                  <TableCell class="text-right"> $450.00 </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <div class="font-medium">Liam Johnson</div>
                    <div class="hidden text-sm text-muted-foreground md:inline">
                      liam@example.com
                    </div>
                  </TableCell>
                  <TableCell class="hidden xl:table-column"> Sale </TableCell>
                  <TableCell class="hidden xl:table-column">
                    <Badge class="text-xs" variant="outline"> Approved </Badge>
                  </TableCell>
                  <TableCell
                    class="hidden md:table-cell lg:hidden xl:table-column"
                  >
                    2023-06-27
                  </TableCell>
                  <TableCell class="text-right"> $550.00 </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </CardContent>
        </Card>
        <Card>
          <CardHeader class="flex flex-row items-center">
            <div class="grid gap-2">
              <CardTitle>Forms</CardTitle>
              <CardDescription> Recent forms</CardDescription>
            </div>
            <Button as-child size="sm" class="ml-auto gap-1">
              <a href="#">
                View All
                <ArrowUpRight class="h-4 w-4" />
              </a>
            </Button>
          </CardHeader>
          <CardContent class="grid gap-8">
            <div class="flex items-center gap-4">
              <Avatar class="hidden h-9 w-9 sm:flex">
                <AvatarImage src="/avatars/01.png" alt="Avatar" />
                <AvatarFallback>OM</AvatarFallback>
              </Avatar>
              <div class="grid gap-1">
                <p class="text-sm font-medium leading-none">Olivia Martin</p>
                <p class="text-sm text-muted-foreground">
                  olivia.martin@email.com
                </p>
              </div>
              <div class="ml-auto font-medium">+$1,999.00</div>
            </div>
            <div class="flex items-center gap-4">
              <Avatar class="hidden h-9 w-9 sm:flex">
                <AvatarImage src="/avatars/02.png" alt="Avatar" />
                <AvatarFallback>JL</AvatarFallback>
              </Avatar>
              <div class="grid gap-1">
                <p class="text-sm font-medium leading-none">Jackson Lee</p>
                <p class="text-sm text-muted-foreground">
                  jackson.lee@email.com
                </p>
              </div>
              <div class="ml-auto font-medium">+$39.00</div>
            </div>
            <div class="flex items-center gap-4">
              <Avatar class="hidden h-9 w-9 sm:flex">
                <AvatarImage src="/avatars/03.png" alt="Avatar" />
                <AvatarFallback>IN</AvatarFallback>
              </Avatar>
              <div class="grid gap-1">
                <p class="text-sm font-medium leading-none">Isabella Nguyen</p>
                <p class="text-sm text-muted-foreground">
                  isabella.nguyen@email.com
                </p>
              </div>
              <div class="ml-auto font-medium">+$299.00</div>
            </div>
            <div class="flex items-center gap-4">
              <Avatar class="hidden h-9 w-9 sm:flex">
                <AvatarImage src="/avatars/04.png" alt="Avatar" />
                <AvatarFallback>WK</AvatarFallback>
              </Avatar>
              <div class="grid gap-1">
                <p class="text-sm font-medium leading-none">William Kim</p>
                <p class="text-sm text-muted-foreground">will@email.com</p>
              </div>
              <div class="ml-auto font-medium">+$99.00</div>
            </div>
            <div class="flex items-center gap-4">
              <Avatar class="hidden h-9 w-9 sm:flex">
                <AvatarImage src="/avatars/05.png" alt="Avatar" />
                <AvatarFallback>SD</AvatarFallback>
              </Avatar>
              <div class="grid gap-1">
                <p class="text-sm font-medium leading-none">Sofia Davis</p>
                <p class="text-sm text-muted-foreground">
                  sofia.davis@email.com
                </p>
              </div>
              <div class="ml-auto font-medium">+$39.00</div>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  </div>
</template>
