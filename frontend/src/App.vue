<template>
    <v-app>
    <v-toolbar app>
        <v-toolbar-title style="width: 300px" class="headline ml-0 pl-3">
        <span>Keyphrase Extraction</span>
        </v-toolbar-title>
        <v-text-field
        flat
        solo-inverted
        hide-details
        prepend-inner-icon="search"
        label="Search"
        class="hidden-sm-and-down"
        v-model="inputValue"
        @keyup.enter="submit"
        @click:prepend-inner="submit"
        ></v-text-field>
        <v-spacer></v-spacer>
    </v-toolbar>
    <v-content>
        <router-view />
    </v-content>
    </v-app>
</template>

<script>
import { mapActions } from 'vuex';

export default {
    name: 'App',
    components: {
    },
    data() {
        return {
            inputValue: '',
        };
    },
    methods: {
        ...mapActions('documents', [
            'search',
        ]),
        submit() {
            this.search(this.inputValue);
        },
    },
};
</script>
