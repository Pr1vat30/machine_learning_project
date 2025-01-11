<script setup lang="ts">
import { Button } from "@/components/ui/button";
import {
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/app/ui/form";
import Toaster from "@/app/ui/toast/Toaster.vue";
import { useToast } from "@/app/ui/toast/use-toast";
import { Textarea } from "@/app/ui/textarea";
import { vAutoAnimate } from "@formkit/auto-animate/vue";

import { toTypedSchema } from "@vee-validate/zod";
import { useForm } from "vee-validate";
import { h } from "vue";
import * as z from "zod";

const { toast } = useToast();

// Schema di validazione per il campo bio
const formSchema = toTypedSchema(
  z.object({
    bio: z
      .string()
      .min(10, "La bio deve essere lunga almeno 10 caratteri")
      .max(300, "La bio non può superare i 300 caratteri"),
  }),
);

const { isFieldDirty, handleSubmit } = useForm({
  validationSchema: formSchema,
  initialValues: {
    bio: "", // Impostiamo il valore iniziale per il campo bio
  },
});

import { useRoute, useRouter } from "vue-router";
const route = useRoute();
const router = useRouter();

const backToHome = () => {
  router.push("/");
};

// Funzione per inviare i dati del modulo al server
const onSubmit = handleSubmit(async (values) => {
  try {
    console.log(JSON.stringify(values));

    const response = await fetch("http://localhost:8080/add_comment", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        id: route.query.id,
        comment: values.bio,
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to submit the form.");
    }

    const responseData = await response.json();
    const { message } = responseData;

    // Mostra il messaggio di successo
    toast({
      title: "Commento aggiunto con successo!",
      description: h(
        "div",
        { class: "mt-2 w-[340px] rounded-md bg-slate-950 p-4" },
        [h("p", { class: "text-white" }, `Message: ${message}`)],
      ),
      duration: 5000, // Puoi personalizzare la durata del toast
    });
  } catch (error) {
    // Gestione errori: mostra un messaggio di errore
    toast({
      title: "Errore durante l'invio del commento",
      description: h(
        "pre",
        { class: "mt-2 w-[340px] rounded-md bg-red-950 p-4" },
        h("code", { class: "text-white" }, error.message),
      ),
      duration: 5000, // Puoi personalizzare la durata del toast
    });
  }
});
</script>

<template>
  <Toaster />
  <section>
    <div class="m-4 flex justify-center">
      <form
        class="w-full space-y-6 max-w-xl bg-white p-8 rounded-lg border bg-card text-card-foreground shadow-sm"
        @submit="onSubmit"
      >
        <div>
          <h1 class="text-3xl">Forms Builder</h1>
        </div>

        <!-- Stampa id, username e bio -->
        <div class="space-y-2">
          <p><strong>ID:</strong> {{ route.query.id }}</p>
          <p><strong>Username:</strong> {{ route.query.username }}</p>
          <p><strong>Bio:</strong> {{ route.query.bio }}</p>
        </div>

        <!-- Campo Bio -->
        <FormField
          v-slot="{ componentField }"
          name="bio"
          :validate-on-blur="!isFieldDirty"
        >
          <FormItem v-auto-animate>
            <FormLabel>Bio</FormLabel>
            <FormControl>
              <Textarea
                placeholder="Tell us a little bit about yourself"
                class="resize-none"
                v-bind="componentField"
              />
            </FormControl>
            <FormDescription>
              You can <span>@mention</span> other users and organizations.
            </FormDescription>
            <FormMessage />
          </FormItem>
        </FormField>

        <Button class="w-full" type="submit">Submit</Button>
        <Button class="w-full" variant="outline" @click="backToHome"
          >Back to home</Button
        >
      </form>
    </div>
  </section>
</template>
