<script setup lang="ts">
import { Badge } from "@/app/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/app/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/app/ui/dropdown-menu";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/app/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/ui/tabs";

import { File, ListFilter, MoreHorizontal, PlusCircle } from "lucide-vue-next";

import { onMounted, ref } from "vue";

// Crea uno stato per i dati da visualizzare nella tabella
const users = ref<any[]>([]);

const enterCode = async () => {
  try {
    // Effettua una richiesta GET all'endpoint
    const response = await fetch(`http://localhost:8080/get-data/`);

    if (response.ok) {
      const data = await response.json();
      console.log("Dati ricevuti:", data.data);

      // Verifica che i dati siano nel formato corretto e salvali
      if (Array.isArray(data.data) && data.data.length > 0) {
        users.value = data.data; // Salva gli utenti nello stato
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

// Carica i dati quando il componente viene montato
onMounted(() => {
  enterCode();
});
</script>

<template>
  <div class="flex min-h-screen w-full flex-col bg-muted/40">
    <div class="flex flex-col sm:gap-4 sm:py-4">
      <main class="grid flex-1 items-start gap-4 p-4 sm:px-6 sm:py-0 md:gap-8">
        <Tabs default-value="all">
          <div class="flex items-center">
            <TabsList>
              <TabsTrigger value="all"> All </TabsTrigger>
              <TabsTrigger value="active"> Active </TabsTrigger>
              <TabsTrigger value="archived" class="hidden sm:flex">
                Archived
              </TabsTrigger>
            </TabsList>
            <div class="ml-auto flex items-center gap-2">
              <DropdownMenu>
                <DropdownMenuTrigger as-child>
                  <Button variant="outline" size="sm" class="h-7 gap-1">
                    <ListFilter class="h-3.5 w-3.5" />
                    <span class="sr-only sm:not-sr-only sm:whitespace-nowrap">
                      Filter
                    </span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuLabel>Filter by</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem checked> Active </DropdownMenuItem>
                  <DropdownMenuItem> Archived </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              <Button size="sm" variant="outline" class="h-7 gap-1">
                <File class="h-3.5 w-3.5" />
                <span class="sr-only sm:not-sr-only sm:whitespace-nowrap">
                  Export
                </span>
              </Button>
              <Button size="sm" class="h-7 gap-1">
                <PlusCircle class="h-3.5 w-3.5" />
                <span class="sr-only sm:not-sr-only sm:whitespace-nowrap">
                  Add Product
                </span>
              </Button>
            </div>
          </div>
          <TabsContent value="all">
            <Card>
              <CardHeader>
                <CardTitle>Users</CardTitle>
                <CardDescription>
                  Manage your users and view their information.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Id</TableHead>
                      <TableHead>Username</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Data</TableHead>
                      <TableHead>Comments</TableHead>
                      <TableHead>Marketing Emails</TableHead>
                      <TableHead>
                        <span class="sr-only">Actions</span>
                      </TableHead>
                    </TableRow>
                  </TableHeader>

                  <TableBody>
                    <!-- Itera sui dati degli utenti e popola la tabella -->
                    <TableRow v-for="user in users" :key="user.id">
                      <TableCell>{{ user.id }}</TableCell>
                      <TableCell>{{ user.username }}</TableCell>
                      <TableCell>
                        <Badge variant="outline">
                          {{ user.status }}
                        </Badge>
                      </TableCell>
                      <TableCell>{{ user.data }}</TableCell>
                      <TableCell>{{ user.numero_commenti }}</TableCell>
                      <TableCell>{{
                        user.marketing_emails ? "Yes" : "No"
                      }}</TableCell>
                      <TableCell>
                        <DropdownMenu>
                          <DropdownMenuTrigger as-child>
                            <Button
                              aria-haspopup="true"
                              size="icon"
                              variant="ghost"
                            >
                              <MoreHorizontal class="h-4 w-4" />
                              <span class="sr-only">Toggle menu</span>
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuLabel>Actions</DropdownMenuLabel>
                            <DropdownMenuItem>Edit</DropdownMenuItem>
                            <DropdownMenuItem>Delete</DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </CardContent>
              <CardFooter>
                <div class="text-xs text-muted-foreground">
                  Showing <strong>1-10</strong> of
                  <strong>{{ users.length }}</strong> users
                </div>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  </div>
</template>
